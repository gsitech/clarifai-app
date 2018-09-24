# clarifai-app
GSI Technology Repository Blog For The Clarifai App

This is an application to perform visual search using Clarifai's API. 

![](images/cover_page.png)

Being by installing the Clarifai Python package:
pip install clarifai

This application queries Clarifai's API to look for images that are visually similar to a query image.
The next step would involve creating a Clarifai account and obtaining an API key.

Create a custom database of images using Clarifai's explorer.
Every time the API is queried, it looks up this custom database for similar images.

Create a new folder and add all query images to it.

Run the python script as:

python Search.py <api_key> <path_to_folder_containing_images>

A user may also enter these into the script manually for convenience of reuse.

For more information visit my [blog](https://medium.com/gsi-technology/getting-started-with-clarifais-sdk-c642be7c45eb)
