import argparse
from lib.scrape_web import scrape_web_bachelor_degrees, scrape_web_campus_pages
from lib.discord_bot import run_discord_bot
from pathlib import Path
from dotenv import load_dotenv

def main():
    results_path = Path("results")
    results_path.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="TM Courses scraper and Discord bot",
        add_help=True
    )
    parser.add_argument(
        "--scrape-tm-courses",
        action="store_true",
        help="Only run the web scraper for TM courses information"
    )
    parser.add_argument(
        "--scrape-tm-info-pages",
        action="store_true",
        help="Only run the web scraper for TM info pages"
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Only run the Discord bot"
    )
    parser.add_argument(
        "--file-path-courses",
        action="store",
        help="Path to the courses data file"
    )
    parser.add_argument(
        "--file-path-info-pages",
        action="store",
        help="Path to the info pages data file"
    )
    
    args = parser.parse_args()
    
    run_scrape_tm_courses = args.scrape_tm_courses
    run_scrape_tm_info_pages = args.scrape_tm_info_pages
    run_bot = args.bot
    data_file_path_courses: str = args.file_path_courses or "courses_by_fase.json"
    data_file_path_info_pages: str = args.file_path_info_pages or "Info_Pages.parquet"

    data_file_path_courses: Path = results_path / data_file_path_courses
    data_file_path_info_pages: Path = results_path / data_file_path_info_pages

    if run_scrape_tm_courses:
        print("Running web scraper for TM courses...")
        scrape_web_bachelor_degrees(results_path=results_path)

    elif run_scrape_tm_info_pages:
        print("Running web scraper for TM info pages...")
        scrape_web_campus_pages(result_path=results_path)

    elif run_bot:
        print("Starting Discord bot...")
        load_dotenv()
        run_discord_bot(
            data_file_path_courses=data_file_path_courses,
            data_file_path_info_pages=data_file_path_info_pages
        )

    else:
        print("No valid arguments provided. Use --help for more information.")

if __name__ == "__main__":
    main()
