import argparse
from lib.scrape_web import scrape_web
from lib.discord_bot import run_discord_bot
from pathlib import Path
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(
        description="TM Courses scraper and Discord bot"
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Only run the web scraper"
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Only run the Discord bot"
    )
    
    args = parser.parse_args()
    
    results_path = Path("results")
    data_file_path = Path("results/courses_by_fase.json")
    
    # If no flags, run both
    run_scrape = args.scrape or (not args.scrape and not args.bot)
    run_bot = args.bot or (not args.scrape and not args.bot)
    
    if run_scrape:
        print("Running web scraper...")
        scrape_web(results_path=results_path)
    
    if run_bot:
        print("Starting Discord bot...")
        load_dotenv()
        run_discord_bot(data_file_path=data_file_path)

if __name__ == "__main__":
    main()
