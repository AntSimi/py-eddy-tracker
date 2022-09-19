import re

from matplotlib.animation import FuncAnimation


class VideoAnimation(FuncAnimation):
    def _repr_html_(self, *args, **kwargs):
        """To get video in html and have a player"""
        content = self.to_html5_video()
        return re.sub(
            r'width="[0-9]*"\sheight="[0-9]*"', 'width="100%" height="100%"', content
        )

    def save(self, *args, **kwargs):
        if args[0].endswith("gif"):
            # In this case gif is used to create thumbnail which is not used but consume same time than video
            # So we create an empty file, to save time
            with open(args[0], "w") as _:
                pass
            return
        return super().save(*args, **kwargs)
