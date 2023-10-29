import praw
import time
def post_to_reddit(image_url,title):
    time.sleep(7)
    reddit = praw.Reddit(
        client_id="hgbzLwGHjRaNJuup7UoYfQ",
        client_secret="FSdaN5gLS8rRj76-ZhV37l8vdx9gyw",
        user_agent="Display1903",
        username='Ill_Independent3989',
        password='PoRoTEPcGRM9VC'
    )
    subreddit_name = 'randomscreenshot'
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
