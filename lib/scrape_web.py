import re
import json
import pprint
import requests
import pprint
import pandas as pd
import numpy as np
import trafilatura
import datetime as dt

from pathlib import Path
from bs4 import BeautifulSoup

from .urls import *
from .embedder import Embedder
from .chunker import Chunker   

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, dt.datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def normalize_channel_name(name: str) -> str:
    # https://support.discord.com/hc/en-us/articles/33694251638295-Discord-Account-Caps-Server-Caps-and-More
    # https://discord.com/developers/docs/resources/channel
    # since most Discord channel names have a max length of 100 characters,
    # we will limit the length to 90 characters to be safe
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "-", name)
    name = "| " + name

    # we want some unicode characters
    # so our categories will look like
    # <fase nr emoji> category name
    # and our channels will look like
    # <course name emoji> | course-name
    # to automate this i suggest to do a post processing step
    # where we add the emojis based on the course name by giving the json file to 
    # a llm (or ssm if available in the future) and asking it to add relevant emojis based on the course name 

    return name[:90]


def scrape_web_bachelor_degrees(results_path: Path) -> Path:
    """
        Scrape the web pages for the bachelor degrees and extract course information.
        Save the results in JSON files in the specified results_path.
        The JSON files will be used later by the Discord bot to create channels , categories and roles.

        Args:
            results_path (Path): Path to save the scraped data.

        Returns:
            the path to the JSON file containing the scraped course data.
    """
    
    global_results = {}
    result_path = results_path
    result_path.mkdir(exist_ok=True)

    for bachelor_degree, url in TM_BACHELOR_DEGREES.items():
        global_results[bachelor_degree] = {}
        print(f"Fetching data from: {url.strip()[:50]}...")

        response = requests.get(url.strip(), timeout=15)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # by taking a look at the HTML structure, we can find the relevant data
        # in divs with class "opo-table-wrapper"
        # inside these divs, each course is represented by a tr with class "module-row"
        # a other piece of data is stored in data-fases attribute of the tr
        # the name of the course is in a td with class "module-title" inside an a tag
        # in a css selector way:
        # div.opo-table-wrapper tr.module-row td.module-title a

        wrappers = soup.find_all("div", class_="opo-table-wrapper")
        results = {}

        if not wrappers:
            print(f"No opo-table-wrapper found for {bachelor_degree}")
            continue

        for wrapper in wrappers:
            module_rows = wrapper.find_all("tr", class_="module-row")
            if not module_rows:
                continue

            for row in module_rows:
                title_td = row.find("td", class_="module-title")
                if not title_td:
                    continue

                link = title_td.find("a")
                if not link:
                    continue

                course_title = link.text.strip()
                fases = row.get("data-fases")

                print(f"Course Title: {course_title}")
                print(f"Fases: {fases}")
                print("-" * 40)
                results[course_title] = fases

        with open(result_path / f"{bachelor_degree}.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        global_results[bachelor_degree] = results


    with open(result_path / "courses.json", "w", encoding="utf-8") as f:
        json.dump(global_results, f, ensure_ascii=False, indent=4)


    # postprocessing by splitting by fase => 1 , 2 , 3
    preprocessed_results = {}
    for bachelor_degree, courses in global_results.items():
        preprocessed_results[bachelor_degree] = {"fase_1": {}, "fase_2": {}, "fase_3": {}}

        pprint.pprint(courses)
        for course_title, fases in courses.items():
            if fases:
                for fase in fases.split(","):
                    fase:str = fase.strip().replace("[","").replace("]","").replace("'","")
                    if f"fase_{fase}" in preprocessed_results[bachelor_degree]:
                        preprocessed_results[bachelor_degree][f"fase_{fase}"][course_title] = [normalize_channel_name(course_title) , fases]

    with open(result_path / "courses_by_fase.json", "w", encoding="utf-8") as f:
        json.dump(preprocessed_results, f, ensure_ascii=False, indent=4)

    return result_path / "courses_by_fase.json"

def scrape_web_campus_pages(result_path: Path) -> Path:
    """
        Scrape the campus pages and return the text content as a list of strings.

        Args:
            result_path (Path): Path to save the scraped data.

        Returns:
            list of strings: the path to the parquet file containing the scraped info pages.

        Note:
            The function saves the scraped data in a Parquet file at the specified result_path.
            A parquet file is chosen for its efficiency in storing tabular data.
            However, a parquet file cannot be easily viewed in a text editor.
            To view the data, you can use pandas to read the parquet file and display its contents
    """
    embedding_model: Embedder = Embedder(
        model_name="ibm-granite/granite-embedding-278m-multilingual"
    )
    chunker = Chunker(
        embedding_model=embedding_model,
        max_chunk_length=500,
    )

    EDU_KEYWORDS = {
        "opleiding", "fase", "studiepunten", "duur",
        "programma", "vakken", "curriculum",
        "bachelor", "graduaat", "campus",
        "stage", "afstudeer", "keuzetraject",
        "credits", "job", "beroep"
    }

    def is_educational(text: str) -> bool:
        lowered = text.lower()
        return any(k in lowered for k in EDU_KEYWORDS)
    
    texts = []
    for url in CAMPUS_PAGE_URL_PAGES:
        print(f"Scraping campus page: {'/'.join(url.split('/')[:-3])}...")
        
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            print(f"Failed to download page: {url}")
            continue
        # we want to extract the main content of the page
        # and so we use trafilatura to do that
        # this will gather the minimal main content from the page
        text = trafilatura.extract(
            downloaded, 
            include_comments=False,
            favor_recall=True, 
            include_tables=True,
            deduplicate=True,
        )
        if text:
            texts.append(text)
        else:
            print(f"No text extracted from page: {url}")

    chunks, eval = chunker.process(texts)

    # removes chunks that are just whitespace
    ws_ = re.compile(r"\s+")
    # removes repeated words
    rpt_ = re.compile(r"\b(\w+)( \1\b)+")
    # removes lines that are just dashes or asterisks
    line_ = re.compile(r"^[-*]{3,}$", re.MULTILINE)
    filtered = [
        ws_.sub(" ", line_.sub("", rpt_.sub(r"\1", p))).strip()
        for p in chunks
        if is_educational(p)
    ]


    print(f"Extracted {len(filtered)} chunks from campus pages.")
    with open(result_path / "Info_Pages_Eval.json", "w", encoding="utf-8") as f:
        json.dump(eval, f, ensure_ascii=False, indent=4, cls=JSONEncoder)

    embeddings = embedding_model.embed(filtered)
    embeddings_df = pd.DataFrame(
        columns=["text", "embedding"], 
        data={"text": filtered, "embedding": embeddings}
    )

    embeddings_df.info()
    embeddings_df.to_parquet(result_path / "Info_Pages.parquet", index=False)

    return result_path / "Info_Pages.parquet"
