import datetime


def _try_parsing_date(text):
    """
    Helper function to try parsing text that either does or does not have a time
    To make the functions below easier, since often time is optional in the apis
    """
    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d']:
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError('No valid date format found')
