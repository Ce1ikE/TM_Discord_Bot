# TM Courses Discord Bot

## Overview

This project scrapes course information from Thomas More bachelor degree programs and campus information pages, then uses a Discord bot with AI capabilities to automatically organize servers and answer questions about campus life using RAG (Retrieval-Augmented Generation).

## Features

### Web Scraping
- **Course Data**: Extracts course information from TM bachelor degree pages (Autotechnologie, Elektromechanica, Elektronica-ICT, Ontwerp en Productietechnologie)
- **Campus Information**: Scrapes TM info pages using trafilatura for clean text extraction
- **Data Processing**: Saves scraped data as JSON and Parquet files with embeddings for RAG

### Discord Bot Capabilities
- **Server Structure Management**: Creates organized categories, channels, and roles based on course structure
  - Categories organized by fase (year) and bachelor degree
  - Individual channels per course with normalized names
  - Emoji-based visual hierarchy
- **AI Question Answering**: RAG-based Q&A system using local LLM
  - Embeds questions and retrieves relevant context from campus documents
  - Uses IBM Granite or Qwen models for text generation
  - Streams responses with Discord rate-limit handling
- **Administrative Commands**:
  - `!build` - Build server structure from scraped data (dry-run mode supported)
  - `!ask <question>` - Ask questions about campus/courses using AI
  - `!statistics` - Display server statistics and bot configuration
  - `!list` - List all categories and channels with mentions
  - `!list_only_pal_channels` - List only PAL-related channels
  - `!list_only_students_channels` - List student-facing channels
  - `!clean_channel` - Remove all bot messages from current channel
  - `!joins_over_time` - Visualize member growth over time
  - `!joins_by_month` - Visualize member joins by month

### AI/ML Components
- **Embeddings**: IBM Granite multilingual embedding models (278M/107M) or sentence-transformers
- **LLM Generation**: Support for IBM Granite, Qwen 2.5, Phi-3.5, TinyLlama models
- **RAG Pipeline**: Cosine similarity-based retrieval with configurable thresholds
- **Document Chunking**: Hybrid chunking with semantic awareness (via Docling)

## Installation

Install dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv pip install -e .
```

## Usage

### Scrape course data only

```bash
uv run python main.py --scrape-tm-courses
```

Fetches course information from TM bachelor degree pages and saves to `results/courses_by_fase.json`.

### Scrape campus info pages only

```bash
uv run python main.py --scrape-tm-info-pages
```

Scrapes TM campus information pages, chunks text, generates embeddings, and saves to `results/Info_Pages.parquet`.

### Run Discord bot only

```bash
uv run python main.py --bot
```

Starts the Discord bot with AI capabilities. Requires `.env` configuration.

### Custom data file paths

```bash
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