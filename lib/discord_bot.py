
# http://realpython.com/how-to-make-a-discord-bot-python/
from importlib.resources import path
import os
import json
import discord
import emoji
import asyncio

from discord.ext import commands
from pathlib import Path
from .urls import TM_BACHELOR_DEGREES




def fase_to_emoji(fase: str) -> str:
    """
    Map fase numbers to specific emojis.

        :param fase: The fase number as a string.
        :return: Corresponding emoji as a string.
    """
    mapping = {
        "1": emoji.emojize(":one:"),
        "2": emoji.emojize(":two:"),
        "3": emoji.emojize(":three:"),
    }
    return mapping.get(fase, emoji.emojize(":question:"))

def bachelor_degree_to_emoji(bachelor_degree: str) -> str:
    """
    Map bachelor degree codes to specific emojis.

        :param bachelor_degree: The code of the bachelor degree.
        :return: Corresponding emoji as a string.
    """
    mapping = {
        "BACHELOR_AUTOTECHNOLOGIE": emoji.emojize(":automobile:"),
        "BACHELOR_ELEKTROMECHANICA": emoji.emojize(":gear:"),
        "BACHELOR_ONTWERP_EN_PRODUCTIETECHNOLOGIE": emoji.emojize(":triangular_ruler:"),
        "BACHELOR_ELEKTRONICA_ICT": emoji.emojize(":computer:"),
    }
    return mapping.get(bachelor_degree, emoji.emojize(":question:"))

def fase_to_year(fase: str) -> str:
    """
    Convert fase numbers to academic year strings.

        :param fase: The fase number as a string.
        :return: Corresponding academic year as a string.
    """
    mapping = {
        "1": "FIRST YEAR",
        "2": "SECOND YEAR",
        "3": "THIRD YEAR",
    }
    return mapping.get(fase, "Unknown Year")

def load_results(filename_path: Path) -> dict:
    """
    Load the scraped results from a JSON file.

        :param filename_path: Path to the JSON file.
        :return: Dictionary with the scraped results.
    """
    with open(filename_path, "r", encoding="utf-8") as f:
        return json.load(f)


async def build_structure(
    guild: discord.Guild,
    ctx: commands.Context,
    data: dict,
    dry_run: bool = True
):
    """
    Build the server structure based on the scraped results.
    Create categories, channels, and roles as needed.

        :param guild: The Discord guild (server) where the structure will be built.
    """
    # a wrapper to avoid destructive operations in dry-run mode
    async def maybe_create(
        action: str, 
        coro,
        **coro_kwargs
    ):
        """
        Wrapper to conditionally execute a coroutine based on DRY_RUN environment variable.
        
            :param action: Description of the action to be performed.
            :param coro: Coroutine to be executed if not in dry-run mode.
        """
        # so depending on the dry-run mode we either execute the action or just print it
        if dry_run:
            await ctx.send(f"`[DRY-RUN]` {action}")
            return
        else:
            await ctx.send(f"`[RUNNING]` {action}")
            return await coro(**coro_kwargs) 
    
    # top level loop over each bachelor degree
    for bachelor_degree, results in data.items():
        bachelor_degree: str
        results: dict

        bachelor_degree_emoji: str = bachelor_degree_to_emoji(bachelor_degree)
        bachelor_degree_display: str = bachelor_degree.replace("BACHELOR_", "").replace("_", " ").title()
        # once we have processed the name we can go to the next
        # step which is to loop over each fase of the bachelor degree 
        for fase, courses in results.items():
            fase: str
            courses: dict
            fase = fase.replace("[", "").replace("]", "").replace("fase_", "")
            fase_emoji: str = fase_to_emoji(fase)
            category_name: str = f"{fase_emoji} | {bachelor_degree_emoji} {bachelor_degree_display} - {fase_to_year(fase)}"
            # creates a category for the fase under the bachelor degree
            category_name = category_name[:90]
            category = await maybe_create(
                # using backticks code blocks for better formatting in Discord
                # nice touch ;)
                action=f"Creating category: #{category_name}",
                coro=guild.create_category,
                name=category_name
            )

            # once we have a category we can loop over each course
            for course_title, course_info in courses.items():
                course_title: str
                course_info: list

                # create a text channel for the course under the category
                channel_name = course_info[0].lower().replace(" ", "-").replace("_", "-")
                # limit to 90 characters to avoid Discord limits
                channel_name = channel_name[:90]
                # slight delay to avoid rate limits
                # cause otherwise we might hit a 503 error from Discord
                # no big deal of course since we can restart the process
                await asyncio.sleep(1)  
                channel = await maybe_create(
                    action=f"Creating channel : #{channel_name} => {category_name}",
                    coro=guild.create_text_channel,
                    name=channel_name,
                    category=category
                )

   

def run_discord_bot(
    data_file_path: Path = Path("results")
):
    """
    Run the Discord bot that creates channels, categories, and roles
    based on the scraped course information.
    """
    
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
    GUILD_ID = int(os.getenv("GUILD_ID"))
    DRY_RUN = os.getenv("DRY_RUN") == "1"

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(
        command_prefix="!", 
        intents=intents
    )

    @bot.event
    async def on_ready():
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

    @bot.command()
    async def test(ctx):
        await ctx.send("**I'm working!**")

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        
        print(f"Message received: {message.content}")
        await bot.process_commands(message) 

    @bot.command()
    async def ping(ctx: commands.Context):
        await ctx.send("Pong!")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list_only_pal_channels(ctx: commands.Context):
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
        guild = ctx.guild
        message = "**Server Structure:**\n"
        category_structure = ""
        for category in guild.categories:
            if "YEAR" in category.name or "GENERAL" in category.name:
                category_structure += f"> **Category:** {category.name}\n"
                for channel in category.channels:
                    category_structure += f">   - Channel: {channel.mention}\n"
                await ctx.send(category_structure)
                category_structure = ""

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def clean_channel(ctx: commands.Context):
        # when invoked as !clean_channel
        # deletes all bot messages in the current channel
        channel = ctx.channel
        channel_name = channel.name

        def is_bot_message(msg):
            return msg.author == bot.user

        deleted = await channel.purge(limit=None, check=is_bot_message)
        await ctx.send(f"Deleted {len(deleted)} messages from #{channel_name}.")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def statistics(ctx: commands.Context):
        guild = ctx.guild
        total_categories = len(guild.categories)
        total_channels = sum(len(category.channels) for category in guild.categories)
        total_roles = len(guild.roles)
        total_persons = len(guild.members)

        stats_message = (
            f"**Server Statistics:**\n"
            f"> Total Categories: {total_categories}\n"
            f"> Total Channels: {total_channels}\n"
            f"> Total Roles: {total_roles}\n"
            f"> Total People: {total_persons}\n"
        )
        await ctx.send(stats_message)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def list(ctx: commands.Context):
        # lists all categories and channels in the server recursively and 
        # outputs a formattted message with a link to each channel
        guild = ctx.guild
        message = "**Server Structure:**\n"
        category_structure = ""
        for category in guild.categories:
            category_structure += f"> **Category:** {category.name}\n"
            for channel in category.channels:
                category_structure += f">   - Channel: {channel.mention}\n"
            # to avoid hitting message length limits in Discord we send the message in chunks
            await ctx.send(category_structure)
            category_structure = ""

    data = load_results(data_file_path)
    # so once this code is run no channels/categories/roles are created
    # only when the !build command is issued by an administrator in the server
    # if the DRY_RUN env variable is set to 1 no changes are made but actions are printed to the console
    @bot.command()
    @commands.has_permissions(administrator=True)
    async def build(ctx: commands.Context):
        await ctx.send("**## Building server structure... this may take a while. ##**")

        if DRY_RUN:
            await ctx.send("> Dry-run mode is enabled. No changes will be made.")

        await build_structure(
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


