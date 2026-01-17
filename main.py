from lib.scrape_web import scrape_web
from lib.discord_bot import run_discord_bot
from pathlib import Path
from dotenv import load_dotenv

def main():
    # scrape_web(results_path=Path("results"))
    load_dotenv(".env.example")
    run_discord_bot(data_file_path=Path("results/courses_by_fase.json"))

if __name__ == "__main__":
    main()
