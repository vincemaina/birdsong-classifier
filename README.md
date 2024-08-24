## Problem reduction

After a while I hit a roadblock. My model was achieving only 40% accuracy, and the project was becoming messy and unmanagable.

At this point I decided to start again, and this time reduce the complexity of the problem.

I decided I'd pick just 10 british birds, that all sound distinct from each other.

I figured this would make it easier for the model to learn the difference between these different birds.

I also decided that dataset (British Birdsong Dataset) I'd been using was not going to be sufficient, as it did not provide enough training data for each bird (88 species with only 3 recordings each, in some of these the bird is only chirping for a few seconds.)

I created a tool that uses the xeno-canto api to get a list of recording ids for a each bird, and then downloaded 50 recordings for each of the 10 birds I'd selected.

## Data cleaning and augmentation

I decided on my first run that I'd see how well the model performed with no data cleaning and augmentation applied. From what I've read online it is not a given that cleaning and augmenting the audio files will improve model accuracy.