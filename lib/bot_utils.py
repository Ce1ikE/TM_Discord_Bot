
import os
import json
import discord
import emoji
import asyncio
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
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


class BotUtils:
    
    @staticmethod
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
    

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def load_results(filename_path: Path) -> dict:
        """
        Load the scraped results from a JSON file.

            :param filename_path: Path to the JSON file.
            :return: Dictionary with the scraped results.
        """
        with open(filename_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
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

            bachelor_degree_emoji: str = BotUtils.bachelor_degree_to_emoji(bachelor_degree)
            bachelor_degree_display: str = bachelor_degree.replace("BACHELOR_", "").replace("_", " ").title()
            # once we have processed the name we can go to the next
            # step which is to loop over each fase of the bachelor degree 
            for fase, courses in results.items():
                fase: str
                courses: dict
                fase = fase.replace("[", "").replace("]", "").replace("fase_", "")
                fase_emoji: str = BotUtils.fase_to_emoji(fase)
                category_name: str = f"{fase_emoji} | {bachelor_degree_emoji} {bachelor_degree_display} - {BotUtils.fase_to_year(fase)}"
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

    