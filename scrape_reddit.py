
import praw
import argparse
import os
import csv
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def scrape_reddit(subreddit_name, query, output_filename):
    # Authenticate with Reddit
    reddit = praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
        username=os.getenv("REDDIT_USERNAME"),
        password=os.getenv("REDDIT_PASSWORD"),
    )

    subreddit = reddit.subreddit(subreddit_name)
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['date', 'title', 'score', 'url', 'comment']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        saved_comments_count = 0
        for submission in subreddit.search(query, sort="new", limit=100):
            submission_date = datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d')
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list():
                # --- Start of filtering logic ---
                if (comment.author is None or
                    "bot" in comment.author.name.lower() or
                    comment.body in ["[deleted]", "[removed]"] or
                    len(comment.body) < 15 or
                    comment.score <= 0):
                    continue
                # --- End of filtering logic ---

                writer.writerow({
                    'date': submission_date,
                    'title': submission.title,
                    'score': submission.score,
                    'url': submission.url,
                    'comment': comment.body
                })
                saved_comments_count += 1

    print(f"Filtered and saved {saved_comments_count} comments to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Reddit for stock discussions.")
    parser.add_argument("subreddit", help="The subreddit to scrape.")
    parser.add_argument("query", help="The search query.")
    parser.add_argument("-o", "--output", default="reliance_reddit_data.csv", help="The output CSV file name.")
    args = parser.parse_args()
    
    scrape_reddit(args.subreddit, args.query, args.output)
