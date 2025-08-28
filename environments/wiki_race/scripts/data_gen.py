# /// script
# dependencies = [
#   "datasets",
#   "huggingface-hub",
#   "beautifulsoup4",
# ]
# ///

"""
Wiki Race Dataset Generator

This script generates training data for the Wiki Race environment by fetching
random Wikipedia article pairs. It respects Wikipedia's robot policy by:
- Including a proper user-agent header
- Adding delays between requests to avoid rate limiting
- Using reasonable timeouts

See: https://w.wiki/4wJS for Wikipedia's robot policy
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from datasets import Dataset

#### Constants ####

RANDOM_ARTICLE_URL = "https://en.wikipedia.org/wiki/Special:Random"
USER_AGENT = (
    "WikiRace-DataGen/1.0 (https://github.com/ljt019/rl-environments; research@wikirace-dataset.com) Research/Bot"
)
REQUEST_DELAY = 1.0  # Delay between requests in seconds

NUM_TO_GENERATE = 2500


###################


def is_article_suitable(title: str, html_content: str) -> bool:
    """
    Check if an article is suitable for Wiki Race (not too niche/problematic).
    
    Args:
        title: Article title
        html_content: Raw HTML content
        
    Returns:
        True if article is suitable, False otherwise
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Filter out disambiguation pages
    if "(disambiguation)" in title.lower():
        return False
        
    # Filter out "List of meanings" type articles
    if title.lower().startswith(("meanings of", "list of meanings")):
        return False
        
    # Filter out very short articles (likely stubs)
    content_div = soup.find("div", {"id": "mw-content-text"})
    if content_div:
        text_content = content_div.get_text()
        if len(text_content.strip()) < 500:  # Less than 500 characters
            return False
    
    # Filter out pages with very few links (dead ends)
    links = []
    if content_div:
        for link in content_div.find_all("a", href=True):
            href = link["href"]
            if href.startswith("/wiki/") and not any(
                skip in href for skip in [
                    "/wiki/Special:", "/wiki/File:", "/wiki/Category:",
                    "/wiki/Template:", "/wiki/Help:", "/wiki/Portal:",
                    "/wiki/Wikipedia:", "/wiki/Talk:"
                ]
            ):
                links.append(href)
    
    if len(links) < 5:  # Too few outbound links
        return False
        
    return True


def extract_article_info(html_content: str) -> Tuple[str, List[str]]:
    """
    Extract article title and list of linked article names from Wikipedia HTML.

    Args:
        html_content: Raw HTML content from a Wikipedia article

    Returns:
        Tuple of (article_title, list_of_linked_articles)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract article title
    title_elem = soup.find("h1", {"id": "firstHeading"})
    if title_elem:
        title = title_elem.get_text().strip()
    else:
        # Fallback to title tag
        title_tag = soup.find("title")
        title = title_tag.get_text().replace(" - Wikipedia", "").strip() if title_tag else "Unknown Article"

    # Extract links to other Wikipedia articles
    links = []
    content_div = soup.find("div", {"id": "mw-content-text"})

    if content_div:
        # Find all links within the main content
        for link in content_div.find_all("a", href=True):
            href = link["href"]
            # Only include links to other Wikipedia articles (not external links)
            if href.startswith("/wiki/") and not any(
                skip in href
                for skip in [
                    "/wiki/Special:",
                    "/wiki/File:",
                    "/wiki/Category:",
                    "/wiki/Template:",
                    "/wiki/Help:",
                    "/wiki/Portal:",
                    "/wiki/Wikipedia:",
                    "/wiki/Talk:",
                ]
            ):
                # Convert URL path to article name
                article_name = href.replace("/wiki/", "").replace("_", " ")
                # Decode URL encoding
                article_name = requests.utils.unquote(article_name)
                if article_name and article_name not in links:
                    links.append(article_name)

    return title, links


def get_random_article_with_links() -> Tuple[str, List[str]]:
    """
    Fetch a random Wikipedia article and extract its title and links.

    Returns:
        Tuple of (article_title, list_of_linked_articles)
    """
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    }

    try:
        response = requests.get(RANDOM_ARTICLE_URL, headers=headers, timeout=10)
        response.raise_for_status()

        title, links = extract_article_info(response.text)

        # Check if article is suitable for Wiki Race
        if not is_article_suitable(title, response.text):
            print(f"Filtered out: '{title}' (disambiguation/stub/few links)")
            time.sleep(REQUEST_DELAY)
            return get_random_article_with_links()

        # Ensure we have at least some links (redundant with filter, but keeping as safety)
        if not links:
            print(f"Warning: Article '{title}' has no links, trying another...")
            time.sleep(REQUEST_DELAY)
            return get_random_article_with_links()

        return title, links

    except requests.exceptions.RequestException as e:
        print(f"Error fetching article: {e}")
        time.sleep(REQUEST_DELAY * 2)  # Longer delay on error
        raise


def generate_random_article_pair() -> Dict[str, str]:
    """
    Generate a completely random article pair for WikiRace.
    Both start and target are chosen independently and randomly.

    As you noted, all Wikipedia articles are theoretically connected through
    the link network, so we don't need to verify connections - that's the
    challenge of WikiRace!

    Returns:
        Dict with 'start_article' and 'target_article' keys
    """
    try:
        # Get completely random start article WITH links
        start_title, start_links = get_random_article_with_links()

        # Add delay between requests
        time.sleep(REQUEST_DELAY)

        # Get completely random target article
        target_title, _ = get_random_article_with_links()

        # Make sure start and target are different
        if start_title == target_title:
            # If they're the same, get a new target
            time.sleep(REQUEST_DELAY)
            target_title, _ = get_random_article_with_links()

        # Format initial game state similar to Battleship
        formatted_links = "\n".join([f"{i + 1}. {link}" for i, link in enumerate(start_links)])
        
        initial_game_state = f"""You are playing Wiki Race. Your goal is to navigate from a starting Wikipedia article to a target article by clicking on links within articles.

Current article: {start_title}
Target article: {target_title}
Step: 0

Available links:
{formatted_links}

Your path so far: {start_title}

Select your next link using <link>NUMBER</link> format."""

        return {
            "question": initial_game_state,
            "info": {
                "start_article": start_title,
                "target_article": target_title,
            }
        }

    except Exception as e:
        print(f"Error generating random pair: {e}")
        time.sleep(REQUEST_DELAY * 2)
        # Retry
        return generate_random_article_pair()


###################


def generate_start_and_target_article_pair():
    """
    Generate a data point for the Wiki Race dataset.
    Uses completely random article pair generation.
    """
    return generate_random_article_pair()


if __name__ == "__main__":
    print(f"Generating {NUM_TO_GENERATE} random Wikipedia article pairs...")
    print("This may take a while due to rate limiting.")

    dataset_data = []
    for i in range(NUM_TO_GENERATE):
        try:
            print(f"Generating pair {i + 1}/{NUM_TO_GENERATE}...")
            pair = generate_start_and_target_article_pair()
            dataset_data.append(pair)
            print(f"  ✓ {pair['start_article']} → {pair['target_article']}")
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
            break
        except Exception as e:
            print(f"  ✗ Error generating pair {i + 1}: {e}")
            continue

    if dataset_data:
        print(f"\nCreating dataset with {len(dataset_data)} pairs...")
        dataset = Dataset.from_list(dataset_data)

        print("Pushing to Hugging Face Hub...")
        dataset.push_to_hub("ljt019/wiki-race-1000")
        print("✓ Dataset uploaded successfully!")
    else:
        print("No data generated. Exiting.")
