import praw
import time
def post_to_reddit(image_url,title):
    time.sleep(7)
    reddit = praw.Reddit(
        client_id="***",
        client_secret="***",
        user_agent="***",
        username='***',
        password='***'
    )
    subreddit_name = '***'
    subreddit = reddit.subreddit(subreddit_name)
    submission = subreddit.submit_image(
        title=title,
        image_path=image_url,
        nsfw=False,
        spoiler=False
    )
    print(f'Image posted successfully! Here is the link: {submission.url}')

with open('data.txt', 'r') as file:
    stored_data = file.readlines()
post_to_reddit(stored_data[0].strip(),stored_data[1].strip())
