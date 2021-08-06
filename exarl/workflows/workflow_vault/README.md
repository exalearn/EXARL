# Workflows

## K. Cosburn:

Added sync and async workflows for A2C. Main difference is that the training occurs after the episode has completed in the case of both workflows. I have also gotten rid of the "yield" when collecting memories and added a function to clear the memory lists at the end of each episode.
