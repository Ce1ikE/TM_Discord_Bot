# TM Courses Discord Bot

## Overview
This project scrapes course information from Thomas More bachelor degree programs and campus information pages, 
then uses a Discord bot with AI capabilities to automatically organize servers and answer questions about campus life 
using RAG (Retrieval-Augmented Generation).

## Installation
Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv pip install -e .
```

## Usage
```bash
### Scrape course data only
# Fetches course information from TM bachelor degree pages and saves to `results/courses_by_fase.json`.
uv run python main.py --scrape-tm-courses
### Scrape campus info pages only
# Scrapes TM campus information pages, chunks text, generates embeddings, and saves to `results/Info_Pages.parquet`.
uv run python main.py --scrape-tm-info-pages
### Run Discord bot only
# Starts the Discord bot with AI capabilities. Requires `.env` configuration.
uv run python main.py --bot
### Custom data file paths
uv run python main.py --bot --file-path-courses custom_courses.json --file-path-info-pages custom_info.parquet
```

## Configuration

Create a `.env` file with:

```env
DISCORD_TOKEN=your_discord_bot_token_here
GUILD_ID=your_discord_server_id
DRY_RUN=1  # Set to 0 to allow actual changes
```

## Bot Commands

All administrative commands require administrator permissions:

### Basic Commands
- `!test` - Test if bot is responsive
- `!ping` - Check bot latency
- `!build` - Build/update server structure from scraped course data

### AI & Questions
- `!tm_ai <question>` - Ask AI about campus/courses (RAG-based, max 200 chars)

### Server Management
- `!list` - List all categories and channels with mentions
- `!list_only_pal_channels` - Filter PAL channels only
- `!list_only_students_channels` - Filter student channels only
- `!clean_channel` - Delete all bot messages in current channel

### Statistics & Analytics
- `!statistics` - Comprehensive server stats (members, channels, roles, boost info)
- `!joins_over_time` - Member growth visualization (cumulative line chart)
- `!joins_by_month` - Member joins by month (bar chart)
- `!member_status` - Member status breakdown (online/idle/dnd/offline) with pie chart
- `!role_distribution` - Top 10 roles by member count (bar chart)
- `!activity_heatmap` - Member join patterns by day/hour (heatmap)
- `!channel_stats` - Channel breakdown by type and category
- `!boost_stats` - Server boost level, progress, and benefits
- `!demographics` - Member demographics (bots vs humans, account ages)