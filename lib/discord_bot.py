
import os
import json
import discord
import emoji
import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# REQUIRED for headless environments
matplotlib.use("Agg")  

from io import BytesIO
from collections import Counter
# http://realpython.com/how-to-make-a-discord-bot-python/
from discord.ext import commands
from pathlib import Path
from threading import Thread
from transformers import TextIteratorStreamer, StoppingCriteriaList, GenerationConfig

from .llm import LLMGenerator
from .embedder import Embedder
from .urls import *



def retrieve_context(
    question_emb: list[float], 
    docs_embd: pd.DataFrame, 
    cosine_similarity_threshold: float = 0.75,
    top_k=5
):
    retrieved = []
    for line in docs_embd.itertuples():
        texts = line.text
        embeddings = line.embedding
        # cos_sim(A, B) = dot(A, B) / (||A|| * ||B||)
        cosine_sim = np.dot(question_emb, embeddings) / (np.linalg.norm(question_emb) * np.linalg.norm(embeddings))
        if cosine_sim >= cosine_similarity_threshold:
            print(f"Retrieved chunk with cosine similarity {cosine_sim:.4f}")
            retrieved.append((cosine_sim, texts))
    retrieved.sort(reverse=True)
    print(f"Total retrieved chunks: {len(retrieved)}")
    return "\n".join(line for _, line in retrieved[:top_k])


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
    data_file_path_courses: Path = Path("results/Traject_<..>.json"),
    data_file_path_info_pages: Path = Path("results/Info_Pages.parquet")
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

    llm_generator = LLMGenerator(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
    )
    stream = TextIteratorStreamer(llm_generator.tokenizer, skip_prompt=True, skip_special_tokens=True)

    embedding_model = Embedder(
        model_name="ibm-granite/granite-embedding-278m-multilingual"
    )

    CAMPUS_DOCS = pd.read_parquet(data_file_path_info_pages)
    print(f"Scraped {len(CAMPUS_DOCS)} documents from campus pages.")


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

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        
        print(f"Message received: {message.content}")
        await bot.process_commands(message) 

    @bot.command()
    async def test(ctx):
        await ctx.send("**I'm working!**")

    @bot.command()
    async def ping(ctx: commands.Context):
        await ctx.send("Pong!")

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def tm_ai(ctx: commands.Context, *, question: str):
        
        if not question:
            await ctx.send("Please provide a question after the command.")
            return
        
        limit = 200
        if len(question) > 200:
            await ctx.send(f"Your question is too long. Please limit it to {limit} characters.")
            return
        
        # run the generation in a separate thread, 
        # so that we can fetch the generated text in a non-blocking way.
        retrieved = retrieve_context(
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
        generation_kwargs = dict(
            context=context,
            prompt=question,
            streamer=stream,
            # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
            generation_config=GenerationConfig(
                max_new_tokens=200,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        )
        thread = Thread(
            target=llm_generator.generate, 
            kwargs=generation_kwargs
        )
        
        thread.start()
        msg = await ctx.send(f"{emoji.emojize(':robot_face:')} Generating...")

        buffer: str = ""
        last_edit = time.monotonic()

        for token in stream:
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

        thread.join()

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
            f"> Bot Dry-Run Mode: {'Enabled' if DRY_RUN else 'Disabled'}\n"
            f"> Current Guild Timezone: {time.tzname[0]}\n"
            f"> Current Server Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
            f"> LLM backend: {llm_generator.model_name}\n"
        )
        await ctx.send(stats_message)

    @bot.command()
    @commands.has_permissions(administrator=True)
    async def joins_over_time(ctx: commands.Context):
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

    data = load_results(data_file_path_courses)
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


