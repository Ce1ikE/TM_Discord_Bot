
import os
import discord
import emoji
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import ast
# REQUIRED for headless environments
matplotlib.use("Agg")  

from typing import Dict, Any
from io import BytesIO
from collections import Counter
from pathlib import Path
from threading import Thread

# http://realpython.com/how-to-make-a-discord-bot-python/
from discord.ext import commands, tasks
from discord.ext.commands.core import Command, CogT


from .llm import LLMGenerator
from .embedder import Embedder
from .urls import *
from .bot_utils import BotUtils

def run_discord_bot(
    data_file_path_courses: Path = Path("results/Traject_<..>.json"),
    data_file_path_info_pages: Path = Path("results/Info_Pages.parquet"),
    no_llm: bool = False
):
    """
    Run the Discord bot that creates channels, categories, and roles
    based on the scraped course information.
    """
    
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GUILD_ID = int(os.getenv("GUILD_ID"))
    DRY_RUN = os.getenv("DRY_RUN") == "1"
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 8192))
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
    SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "You are a helpful assistant for students of the Thomas More Campus De Nayer. Use the provided context to answer the question. If you don't know the answer, say you don't know. Always use all the relevant information from the context to provide a complete and accurate answer.")

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    intents.guilds = True
    bot = commands.Bot(
        command_prefix="!", 
        intents=intents
    )

    if not no_llm:
        print("Running bot with LLM. The tm_ai command will work.")
        llm_generator = LLMGenerator(
            model_name="unsloth/gemma-4-E2B-it-GGUF:gemma-4-E2B-it-Q4_0.gguf",
            max_context_tokens=MAX_CONTEXT_TOKENS,
            max_new_tokens=MAX_NEW_TOKENS,
            system_prompt=SYSTEM_PROMPT
        )

        embedding_model = Embedder(
            model_name="ibm-granite/granite-embedding-278m-multilingual"
        )

        CAMPUS_DOCS = pd.read_parquet(data_file_path_info_pages)
        print(f"Scraped {len(CAMPUS_DOCS)} documents from campus pages.")
    else:
        llm_generator = None
        embedding_model = None
        CAMPUS_DOCS = None
        print("Running bot without LLM. The tm_ai command will respond with a service offline message.")

    @bot.event
    async def on_ready():
        """
        Event handler for when the bot has successfully connected to Discord.
        """
        print(f"Logged in as {bot.user}")

        guild = bot.get_guild(GUILD_ID)
        if guild is None:
            print(f"Guild with ID {GUILD_ID} not found.")
            await bot.close()
            return

        # permissions check to avoid destructive operations 
        # without proper rights

        me = guild.me
        perms = me.guild_permissions

        assert perms.manage_channels, "Bot lacks manage_channels permission"
        assert perms.manage_roles, "Bot lacks manage_roles permission"

        print(f"Connected to guild: {guild.name}")
        print(f"Dry-run mode: {DRY_RUN}")

    @bot.event
    async def on_message(message: discord.Message):
        """
        Event handler for incoming messages. 
        We need to process commands in order for the bot to respond to them.
        """
        if message.author == bot.user:
            return
        
        print(f"Message received: {message.content}")
        await bot.process_commands(message) 
        print(f"Finished processing message: {message.content}")

    @bot.command()
    async def test(ctx: commands.Context):
        """
        Simple command to test if the bot is working.
        """
        await ctx.send("**I'm working!**")

    @bot.command()
    async def ping(ctx: commands.Context):
        """
        Simple command to test if the bot is responsive.
        """
        await ctx.send("Pong!")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def print_full_commands(ctx: commands.Context):
        """
        list all available commands and their descriptions.
        (ADMIN version)
        """
        for cmd in sorted(bot.commands, key=lambda c: c.name):
            try:
                if cmd.name == "run_discord_bot":
                    continue
                doc = cmd.help or getattr(cmd.callback, "__doc__", None) or "No description"
                await ctx.send(f"**!{cmd.name}**: {doc}")
            except Exception:
                continue

    @bot.command()
    async def print_commands(ctx: commands.Context):
        """
        list all available commands and their descriptions.
        (USER version)
        """
        for cmd in sorted(bot.commands, key=lambda c: c.name):
            try:
                if await cmd.can_run(ctx):
                    doc = cmd.help or getattr(cmd.callback, "__doc__", None) or "No description"
                    await ctx.send(f"**!{cmd.name}**: {doc}")
            except Exception:
                # skip commands that raise permission checks or other errors
                continue
    

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def tm_ai(ctx: commands.Context, *, question: str):
        """
        Answer questions about the Thomas More Campus De Nayer using the LLM and retrieved context 
        from the scraped info pages.
        """

        if no_llm:
            await ctx.send("Service currently offline. Please try again later.")
            return
        
        if not question:
            await ctx.send("Please provide a question after the command.")
            return
        
        # run the generation in a separate thread, 
        # so that we can fetch the generated text in a non-blocking way.
        retrieved = BotUtils.retrieve_context(
            question_emb=embedding_model.embed([question])[0],
            docs_embd=CAMPUS_DOCS,
            cosine_similarity_threshold=0.75,
            top_k=2
        )

        context = f"""
        Je bent iPAL, een AI-assistent voor Thomas More Campus De Nayer.
        Gebruik ALLEEN de onderstaande informatie om te antwoorden.
        Als het antwoord niet aanwezig is, antwoord dan precies: "Ik weet het niet."

        Extra Informatie:
        {retrieved}
        
        
        Regels:
        - Beantwoord ALLEEN de vraag van de gebruiker.
        - Stel GEEN vervolgvragen.
        - Voeg GEEN meerdere vraag-antwoordparen toe.
        - Ga NIET verder met het gesprek.
        - Houd antwoorden feitelijk, kort en specifiek.
        - Als het antwoord onbekend is of niet in je kennis zit, antwoord dan precies: "Ik weet het niet."
        - Verzin GEEN feiten.
        - Vermeld GEEN URL's, tenzij hier expliciet om wordt gevraagd.
        
        U vertegenwoordigt Thomas More Campus De Nayer.

        """

        msg = await ctx.send(f"{emoji.emojize(':robot_face:')} Generating...")

        buffer: str = ""
        last_edit = time.monotonic()

        for token in llm_generator.generate(context=context, prompt=question, stream=True):
            buffer += token
            # because Discord has rate limits we only edit the message
            # once every second to avoid hitting those limits
            # we also have to "edit" the message instead of sending a new one
            # to avoid spamming the channel with messages and because Discord has no streaming API
            if time.monotonic() - last_edit > 1.0:
                await msg.edit(
                    content=(
                        f"{emoji.emojize(':robot_face:')}"
                        f"**Question**\n"
                        f"> {question}\n\n"
                        f"**Answer**\n"
                        f"{buffer.strip()}"
                    )
                )
                last_edit = time.monotonic()

        await msg.edit(
            content=(
                f"{emoji.emojize(':robot_face:')}"
                f"**Question**\n"
                f"> {question}\n\n"
                f"**Answer**\n"
                f"{buffer.strip()}"
            )
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list_only_pal_channels(ctx: commands.Context):
        """
        List only the channels related to PAL.
        This can be useful to get an overview of the channels that are relevant
        for PAL and to check if they are properly organized.
        """
        guild = ctx.guild
        message = "**Server Structure:**\n"
        category_structure = ""
        for category in guild.categories:
            if "PAL" in category.name:
                category_structure += f"> **Category:** {category.name}\n"
                for channel in category.channels:
                    category_structure += f">   - Channel: {channel.mention}\n"
                await ctx.send(category_structure)
                category_structure = ""

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list_only_students_channels(ctx: commands.Context):
        """
        List only the channels related to students. 
        This can be useful to get an overview of the channels that are relevant 
        for students and to check if they are properly organized.
        """
        guild = ctx.guild
        msg = await ctx.send("**Server Structure:**\n")
        last_edit = time.monotonic()

        category_structure = ""
        for category in guild.categories:
            if "YEAR" in category.name or "GENERAL" in category.name:
                category_structure += f"> **Category:** {category.name}\n"
                for channel in category.channels:
                    category_structure += f">   - Channel: {channel.mention}\n"
                
                if time.monotonic() - last_edit > 1.0 or len(category_structure) > 1500:
                    await msg.edit(content=category_structure)
                    last_edit = time.monotonic()

        await msg.edit(content=category_structure)


    @bot.command()
    @commands.has_permissions(administrator=True)
    async def clean_channel(ctx: commands.Context):
        """
        Clean the current channel by deleting all messages sent by the bot. 
        This can be useful to remove old bot messages and keep the channel tidy. 
        Use with caution as this will permanently delete messages.
        """
        # when invoked as !clean_channel
        # deletes all bot messages in the current channel
        channel = ctx.channel
        channel_name = channel.name

        def is_bot_message(msg: discord.Message):
            return msg.author == bot.user

        if DRY_RUN:
            # count matching messages without deleting
            count = 0
            async for msg in channel.history(limit=None):
                if is_bot_message(msg):
                    count += 1
            await ctx.send(f"`[DRY-RUN]` Would delete {count} messages from #{channel_name}.")
        else:
            deleted = await channel.purge(limit=None, check=is_bot_message)
            await ctx.send(f"Deleted {len(deleted)} messages from #{channel_name}.")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def clean_user_messages(ctx: commands.Context, user: discord.Member):
        """
        Clean messages sent by a specific user in the current channel. 
        This can be useful to remove old messages from a user and keep the channel tidy. 
        Use with caution as this will permanently delete messages.
        """
        channel = ctx.channel
        channel_name = channel.name

        def is_user_message(msg: discord.Message):
            return msg.author == user

        if DRY_RUN:
            count = 0
            try:
                async for msg in channel.history(limit=None):
                    if is_user_message(msg):
                        count += 1
            except discord.Forbidden as e:
                await ctx.send(f"Missing permissions to read messages in #{channel_name}. Cannot perform dry-run count.")
                return
            await ctx.send(f"`[DRY-RUN]` Would delete {count} messages from {user.mention} in #{channel_name}.")
        else:
            deleted = await channel.purge(limit=None, check=is_user_message)
            await ctx.send(f"Deleted {len(deleted)} messages from {user.mention} in #{channel_name}.")        


    @bot.command()
    @commands.has_permissions(administrator=True)
    async def clean_all_user_messages(ctx: commands.Context, user: discord.Member):
        """
        Clean all messages in all channels sent by a specific user.
        This can be usefull to remove all messages from a user across the server, 
        for example in case of a user leaving the school and wanting to remove their data from the server.
        or when someone spams the server and you want to remove all their messages.
        """
        guild = ctx.guild
        await ctx.send("**Cleaning user messages:**\n")

        if DRY_RUN:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    count = 0
                    try:
                        async for msg in channel.history(limit=None):
                            if msg.author == user:
                                count += 1
                    except discord.Forbidden as e:
                        await ctx.send(f"Missing permissions to read messages in #{channel.name}. Skipping.")
                        continue
                    
                    if count:
                        await ctx.send(f"`[DRY-RUN]` Would delete {count} messages from {user.mention} in #{channel.name}.")
        else:
            for channel in guild.channels:
                if isinstance(channel, discord.TextChannel):
                    try:
                        deleted = await channel.purge(limit=None, check=lambda msg: msg.author == user)
                    except discord.Forbidden as e:
                        await ctx.send(f"Missing permissions to read messages in #{channel.name}. Skipping.")
                        continue
                    
                    if deleted:
                        await ctx.send(f"Deleted {len(deleted)} messages from {user.mention} in #{channel.name}.")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def statistics(ctx: commands.Context):
        """
        Show various server statistics and insights. 
        This command provides an overview of the server's structure, member activity, and other relevant information 
        that can help administrators understand their community better.
        """
        guild = ctx.guild
        total_categories = len(guild.categories)
        total_channels = sum(len(category.channels) for category in guild.categories)
        total_text_channels = len([c for c in guild.channels if isinstance(c, discord.TextChannel)])
        total_voice_channels = len([c for c in guild.channels if isinstance(c, discord.VoiceChannel)])
        total_roles = len(guild.roles)
        total_members = len(guild.members)
        online_members = sum(1 for m in guild.members if m.status != discord.Status.offline)
        bot_count = sum(1 for m in guild.members if m.bot)
        human_count = total_members - bot_count
        
        # Calculate server boost info
        boost_level = guild.premium_tier
        boost_count = guild.premium_subscription_count or 0
        
        # Get creation date
        server_age = (discord.utils.utcnow() - guild.created_at).days

        stats_message = (
            f"**{emoji.emojize(':bar_chart:')} Server Statistics:**\n"
            f"> **Guild:** {guild.name}\n"
            f"> **Created:** {guild.created_at.strftime('%Y-%m-%d')} ({server_age} days ago)\n"
            f"> **Owner:** {guild.owner.mention if guild.owner else 'Unknown'}\n\n"
            f"**{emoji.emojize(':busts_in_silhouette:')} Members:**\n"
            f"> Total: {total_members} ({human_count} humans, {bot_count} bots)\n"
            f"> Online: {online_members}\n\n"
            f"**{emoji.emojize(':file_folder:')} Channels:**\n"
            f"> Categories: {total_categories}\n"
            f"> Text Channels: {total_text_channels}\n"
            f"> Voice Channels: {total_voice_channels}\n"
            f"> Total: {total_channels}\n\n"
            f"**{emoji.emojize(':performing_arts:')} Roles:** {total_roles}\n"
            f"**{emoji.emojize(':rocket:')} Boost Level:** {boost_level} ({boost_count} boosts)\n\n"
            f"**{emoji.emojize(':robot:')} Bot Info:**\n"
            f"> Dry-Run Mode: {f'{emoji.emojize(":white_check_mark:")} Enabled' if DRY_RUN else f' {emoji.emojize(":x:")} Disabled'}\n"
            f"> LLM Backend: `{llm_generator.model_name if llm_generator else 'Not available'}`\n"
            f"> Server Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
        )
        await ctx.send(stats_message)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def joins_over_time(ctx: commands.Context):
        """
        Show a line chart of member joins over time (by month).
        """
        FONTDICT = {
            'fontfamily': 'monospace',
            'fontsize': 12,
            'fontweight': 'bold'
        }

        guild = ctx.guild

        join_dates = [
            member.joined_at.date()
            for member in guild.members
            if member.joined_at is not None
        ]

        if not join_dates:
            await ctx.send("No join data available.")
            return

        counts = Counter(join_dates)
        dates = sorted(counts.keys())

        cumulative = []
        total = 0
        for d in dates:
            total += counts[d]
            cumulative.append(total)

        plt.figure(figsize=(10, 5))
        plt.plot(dates, cumulative)
        plt.xlabel("Date", fontdict=FONTDICT)
        plt.ylabel("Total members", fontdict=FONTDICT)
        plt.title("Server member growth over time", fontdict=FONTDICT)
        plt.tight_layout()
        plt.grid(
            visible=True,
            which='both',
            axis='both',
            color='gray',
            linestyle='--',
            linewidth=0.5
        )
        plt.gca().spines[['right', 'top']].set_visible(False)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        # send to Discord
        await ctx.send(
            content="**Member joins over time**",
            file=discord.File(fp=buffer, filename="joins_over_time.png")
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def joins_by_month(ctx: commands.Context):
        """
        Show the number of new members by month with a bar chart.
        """
        FONTDICT = {
            'fontfamily': 'monospace',
            'fontsize': 12,
            'fontweight': 'bold'
        }

        guild = ctx.guild

        join_dates = [
            member.joined_at.strftime("%Y-%m")
            for member in guild.members
            if member.joined_at is not None
        ]

        if not join_dates:
            await ctx.send("No join data available.")
            return

        counts = Counter(join_dates)
        months = sorted(counts.keys())
        values = [counts[m] for m in months]

        plt.figure(figsize=(10, 5))
        plt.bar(months, values)
        plt.xlabel("Month", fontdict=FONTDICT)
        plt.ylabel("New members", fontdict=FONTDICT)
        plt.title("New members by month", fontdict=FONTDICT)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(
            visible=True,
            which='both',
            axis='y',
            color='gray',
            linestyle='--',
            linewidth=0.5
        )
        plt.gca().spines[['right', 'top']].set_visible(False)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        # send to Discord
        await ctx.send(
            content="**Member joins by month**",
            file=discord.File(fp=buffer, filename="joins_by_month.png")
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def member_status(ctx: commands.Context):
        """
        Show breakdown of member statuses with a pie chart.
        """
        FONTDICT = {
            'fontfamily': 'monospace',
            'fontsize': 12,
            'fontweight': 'bold'
        }

        guild = ctx.guild
        
        status_counts = {
            'Online': sum(1 for m in guild.members if m.status == discord.Status.online),
            'Idle': sum(1 for m in guild.members if m.status == discord.Status.idle),
            'Do Not Disturb': sum(1 for m in guild.members if m.status == discord.Status.dnd),
            'Offline': sum(1 for m in guild.members if m.status == discord.Status.offline),
        }
        
        # Filter out zero counts
        status_counts = {k: v for k, v in status_counts.items() if v > 0}
        
        if not status_counts:
            await ctx.send("No status data available.")
            return

        colors = ['#43b581', '#faa61a', '#f04747', '#747f8d']
        plt.figure(figsize=(10, 7))
        plt.pie(
            status_counts.values(), 
            labels=status_counts.keys(), 
            autopct='%1.1f%%',
            colors=colors[:len(status_counts)],
            startangle=90
        )
        plt.title("Member Status Distribution", fontdict=FONTDICT)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        await ctx.send(
            content="**Member Status Breakdown**",
            file=discord.File(fp=buffer, filename="member_status.png")
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def role_distribution(ctx: commands.Context):
        """
        Show the top 10 most common roles in the server.
        """
        FONTDICT = {
            'fontfamily': 'monospace',
            'fontsize': 12,
            'fontweight': 'bold'
        }

        guild = ctx.guild
        
        # Count members per role (excluding @everyone)
        role_counts = {}
        for role in guild.roles:
            if role.name != "@everyone" and len(role.members) > 0:
                role_counts[role.name] = len(role.members)
        
        if not role_counts:
            await ctx.send("No role data available.")
            return
        
        # Sort and get top 10
        sorted_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        role_names = [r[0] for r in sorted_roles]
        role_values = [r[1] for r in sorted_roles]

        plt.figure(figsize=(12, 6))
        plt.barh(role_names, role_values, color='#5865F2')
        plt.xlabel("Number of Members", fontdict=FONTDICT)
        plt.ylabel("Role", fontdict=FONTDICT)
        plt.title("Top 10 Roles by Member Count", fontdict=FONTDICT)
        plt.tight_layout()
        plt.grid(
            visible=True,
            which='both',
            axis='x',
            color='gray',
            linestyle='--',
            linewidth=0.5
        )
        plt.gca().spines[['right', 'top']].set_visible(False)

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        await ctx.send(
            content="**Role Distribution**",
            file=discord.File(fp=buffer, filename="role_distribution.png")
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def activity_heatmap(ctx: commands.Context):
        """
        Show when members joined by day of week and hour.
        """
        FONTDICT = {
            'fontfamily': 'monospace',
            'fontsize': 10,
            'fontweight': 'bold'
        }

        guild = ctx.guild
        
        join_times = [
            (m.joined_at.weekday(), m.joined_at.hour)
            for m in guild.members
            if m.joined_at is not None
        ]
        
        if not join_times:
            await ctx.send("No join time data available.")
            return

        # Create 7x24 heatmap
        heatmap = np.zeros((7, 24))
        for day, hour in join_times:
            heatmap[day][hour] += 1

        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        plt.figure(figsize=(14, 6))
        plt.imshow(heatmap, cmap='YlOrRd', aspect='auto')
        plt.colorbar(label='Member Joins')
        plt.xlabel("Hour of Day", fontdict=FONTDICT)
        plt.ylabel("Day of Week", fontdict=FONTDICT)
        plt.title("Member Join Activity Heatmap", fontdict=FONTDICT)
        plt.xticks(range(24), range(24))
        plt.yticks(range(7), days)
        plt.tight_layout()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()
        buffer.seek(0)

        await ctx.send(
            content="**Activity Heatmap**",
            file=discord.File(fp=buffer, filename="activity_heatmap.png")
        )

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def channel_stats(ctx: commands.Context):
        """
        Show channel statistics breakdown.
        """
        guild = ctx.guild
        
        text_channels = [c for c in guild.channels if isinstance(c, discord.TextChannel)]
        voice_channels = [c for c in guild.channels if isinstance(c, discord.VoiceChannel)]
        categories = guild.categories
        
        # Calculate channels per category
        category_sizes = {}
        for cat in categories:
            category_sizes[cat.name] = len(cat.channels)
        
        # Sort by size
        sorted_cats = sorted(category_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats = (
            f"**{emoji.emojize(':file_folder:')} Channel Statistics**\n\n"
            f"**Channel Types:**\n"
            f"> Text Channels: {len(text_channels)}\n"
            f"> Voice Channels: {len(voice_channels)}\n"
            f"> Categories: {len(categories)}\n\n"
            f"**Top Categories by Channel Count:**\n"
        )
        
        for cat_name, count in sorted_cats:
            stats += f"> {cat_name}: {count} channels\n"
        
        await ctx.send(stats)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def boost_stats(ctx: commands.Context):
        """
        Show server boost statistics.
        """
        guild = ctx.guild
        
        boost_level = guild.premium_tier
        boost_count = guild.premium_subscription_count or 0
        boosters = guild.premium_subscribers
        
        # Boost level thresholds
        next_level_boosts = {0: 2, 1: 7, 2: 14, 3: None}
        next_threshold = next_level_boosts.get(boost_level)
        
        stats = (
            f"**{emoji.emojize(':rocket:')} Server Boost Statistics**\n\n"
            f"> Current Level: **{boost_level}**\n"
            f"> Total Boosts: **{boost_count}**\n"
            f"> Active Boosters: **{len(boosters)}**\n"
        )
        
        if next_threshold:
            remaining = next_threshold - boost_count
            stats += f"> Boosts to Level {boost_level + 1}: **{remaining}**\n"
        else:
            stats += f"> {emoji.emojize(':tada:')} **MAX LEVEL REACHED!**\n"
        
        stats += "\n**Level Benefits:**\n"
        if boost_level >= 1:
            stats += f"> {emoji.emojize(':white_check_mark:')} 128 Kbps audio\n> {emoji.emojize(':white_check_mark:')} Custom server invite background\n> {emoji.emojize(':white_check_mark:')} 50 emoji slots\n"
        if boost_level >= 2:
            stats += f"> {emoji.emojize(':white_check_mark:')} 256 Kbps audio\n> {emoji.emojize(':white_check_mark:')} Server banner\n> {emoji.emojize(':white_check_mark:')} 150 emoji slots\n"
        if boost_level >= 3:
            stats += f"> {emoji.emojize(':white_check_mark:')} 384 Kbps audio\n> {emoji.emojize(':white_check_mark:')} Vanity URL\n> {emoji.emojize(':white_check_mark:')} 250 emoji slots\n"
        if boost_level == 0:
            stats += f"> {emoji.emojize(':x:')} No boost benefits yet. Boost the server to unlock perks!\n"

        await ctx.send(stats)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def demographics(ctx: commands.Context):
        """
        Show member demographics (bots vs humans, account ages).
        """
        guild = ctx.guild
        
        total_members = len(guild.members)
        bot_count = sum(1 for m in guild.members if m.bot)
        human_count = total_members - bot_count
        
        # Calculate account ages
        now = discord.utils.utcnow()
        account_ages = []
        for member in guild.members:
            if not member.bot:
                age_days = (now - member.created_at).days
                account_ages.append(age_days)
        
        avg_age = sum(account_ages) / len(account_ages) if account_ages else 0
        
        # Age categories
        new_accounts = sum(1 for age in account_ages if age < 30)  # < 1 month
        young_accounts = sum(1 for age in account_ages if 30 <= age < 365)  # 1 month - 1 year
        mature_accounts = sum(1 for age in account_ages if age >= 365)  # > 1 year
        
        stats = (
            f"**{emoji.emojize(':busts_in_silhouette:')} Server Demographics**\n\n"
            f"**Member Types:**\n"
            f"> Humans: {human_count} ({human_count/total_members*100:.1f}%)\n"
            f"> Bots: {bot_count} ({bot_count/total_members*100:.1f}%)\n\n"
            f"**Account Ages (Humans Only):**\n"
            f"> Average Age: {avg_age:.0f} days ({avg_age/365:.1f} years)\n"
            f"> New (< 1 month): {new_accounts}\n"
            f"> Young (1 month - 1 year): {young_accounts}\n"
            f"> Mature (> 1 year): {mature_accounts}\n"
        )
        
        await ctx.send(stats)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list(ctx: commands.Context):
        """
        List all categories and channels in the server.
        """
        # lists all categories and channels in the server recursively and 
        # outputs a formattted message with a link to each channel
        guild = ctx.guild
        await ctx.send("**Server Structure:**\n")
        category_structure = ""
        for category in guild.categories:
            category_structure += f"> **Category:** {category.name}\n"
            for channel in category.channels:
                category_structure += f">   - Channel: {channel.mention}\n"
            # to avoid hitting message length limits in Discord we send the message in chunks
            await ctx.send(category_structure)
            category_structure = ""

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list_roles(ctx: commands.Context):
        """
        List all roles in the server.
        """
        guild = ctx.guild
        await ctx.send("**Server Roles:**\n")
        for role in guild.roles:
            if role.name == "@everyone":
                continue
            await ctx.send(f"> {role.name}\n")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list_roles_view(ctx: commands.Context):
        """
        List all roles and what channles they can access.
        """
        guild = ctx.guild
        await ctx.send("**Server Roles and Permissions:**\n")
        
        for role in guild.roles:
            if role.name == "@everyone":
                continue
            await ctx.send(f"> **{role.name}**: ")
            
            for channel in guild.channels:
                perms = channel.permissions_for(role)
                if perms.read_messages:
                    await ctx.send(f">   - Channel: {channel.mention}\n")
            

    # so once this code is run no channels/categories/roles are created
    # only when the !build command is issued by an administrator in the server
    # if the DRY_RUN env variable is set to 1 no changes are made but actions are printed to the console
    data = BotUtils.load_results(data_file_path_courses)
    @bot.command()
    @commands.has_permissions(administrator=True)
    async def build(ctx: commands.Context):
        """
        Build the server structure based on the scraped course information. 
        This command should be used with caution as it can create a lot of channels, categories, and roles. 
        It's recommended to run this in dry-run mode first to see what changes would be made without actually applying them.
        """
        await ctx.send("**## Building server structure... this may take a while. ##**")

        if DRY_RUN:
            await ctx.send("> Dry-run mode is enabled. No changes will be made.")

        await BotUtils.build_structure(
            guild=ctx.guild,
            ctx=ctx, 
            data=data, 
            dry_run=DRY_RUN
        )

        if DRY_RUN:
            await ctx.send("> Dry-run complete. No changes were made.")
        else:
            await ctx.send("> Server structure build complete.")

    bot.run(DISCORD_TOKEN)


