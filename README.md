# TM Courses Discord Bot

## Overview

This project scrapes course information from Thomas More bachelor degree programs and creates a Discord bot that automatically generates server channels, categories, and roles based on the course structure.

## Features

- **Web Scraping**: Extracts course data from TM bachelor degree web pages (Autotechnologie, Elektromechanica, Elektronica-ICT, Ontwerp en Productietechnologie)
- **Discord Bot**: Creates organized server structure with categories per fase (year), channels per course, and assigns roles automatically
- **JSON Export**: Saves scraped data to JSON files for later use

## Installation

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .
```

## Usage

### Scrape course data only

```bash
uv run python main.py --scrape
```

This fetches course information from TM websites and saves it to the `results/` directory.

### Run Discord bot only

```bash
uv run python main.py --bot
```

Requires a `.env` file with your Discord bot token. The bot uses the scraped data to build the server structure.

### Run both (default)

```bash
uv run python main.py
```

Scrapes data first, then starts the Discord bot.

## Configuration

Create a `.env` file with:

```
DISCORD_TOKEN=your_discord_bot_token_here
```
