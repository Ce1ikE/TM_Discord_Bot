import re
import json
import pprint
import requests

from pathlib import Path
from bs4 import BeautifulSoup
from .urls import TM_BACHELOR_DEGREES

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


    return name[:90]


def scrape_web(results_path: Path):
    """
    Scrape the web pages for the bachelor degrees and extract course information.
    Save the results in JSON files in the specified results_path.
    the JSON files will be used later by the Discord bot to create channels , categories and roles.
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
