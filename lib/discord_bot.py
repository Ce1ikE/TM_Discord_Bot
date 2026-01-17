
# http://realpython.com/how-to-make-a-discord-bot-python/
from importlib.resources import path
import os
import json
import discord
import emoji

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
            await ctx.send(f"[DRY-RUN] {action}")
            return
        else:
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
            category_name: str = f"Fase {fase_emoji} |  {bachelor_degree_display} {bachelor_degree_emoji}"
            # creates a category for the fase under the bachelor degree
            category_name = category_name[:90]
            category = await maybe_create(
                action=f"Creating category: {category_name}",
                coro=guild.create_category,
                name=category_name
            )
            
            # creates a role for the category
            role_name = f"Fase-{fase}"
            role = await maybe_create(
                action=f"Creating role: {role_name}",
                coro=guild.create_role,
                name=role_name
            )

            # once we have a category we can loop over each course
            for course_title, course_info in courses.items():
                course_title: str
                course_info: list

                # create a text channel for the course under the category
                channel_name = course_title[:90].lower().replace(" ", "-")
                channel = await maybe_create(
                    action=f"Creating channel: {channel_name} in category: {bachelor_degree_display}",
                    coro=guild.create_text_channel,
                    name=channel_name,
                    category=category
                )

                if role and channel:
                    # optionally, sets permissions for the role on the channel
                    await maybe_create(
                        action=f"Setting permissions for role: {role_name} on channel: {channel_name}",
                        coro=channel.set_permissions,
                        role=role,
                        read_messages=True,
                        send_messages=True
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
        await ctx.send("I'm working!")

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return
        
        print(f"Message received: {message.content}")
        await bot.process_commands(message) 

    @bot.command()
    async def ping(ctx: commands.Context):
        await ctx.send("Pong!")

    data = load_results(data_file_path)
    # so once this code is run no channels/categories/roles are created
    # only when the !build command is issued by an administrator in the server
    # if the DRY_RUN env variable is set to 1 no changes are made but actions are printed to the console
    @bot.command()
    @commands.has_permissions(administrator=True)
    async def build(ctx: commands.Context):
        await ctx.send("Building server structure... this may take a while.")

        if DRY_RUN:
            await ctx.send("Dry-run mode is enabled. No changes will be made.")

        await build_structure(
            guild=ctx.guild,
            ctx=ctx, 
            data=data, 
            dry_run=DRY_RUN
        )

        if DRY_RUN:
            await ctx.send("Dry-run complete. No changes were made.")
        else:
            await ctx.send("Server structure build complete.")

    bot.run(DISCORD_TOKEN)


