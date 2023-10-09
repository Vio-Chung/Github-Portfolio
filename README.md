[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/9JYWXZji)
# MPATE-GE 2623 - Music Information Retrieval
## Homework 1

**Instructions:**

1. Complete parts 1 through 5, filling in code in the `utils.py` file where indicated **# YOUR CODE HERE** or responses in `this notebook` where marked with **# YOUR RESPONSE HERE** or **# YOUR CODE HERE**.
2. **Document** your code. Add comments explaining what the different parts of your code are doing.
3. If you copy code from external resources (e.g. librosa's examples), include references as comments.
4. When finished, commit and push this completed notebook file along with the `utils.py` file to your GitHub repository corresponding to this homework.
5. IMPORTANT: do not modify any of the provided code.

**How to work with the `utils.py` file and Google Colab:**

You can run your code remotely with Google Colab if you add the `utils.py` file to your files' folder (search for the folder icon in the menu on the left). But **CAREFUL**, you should copy any changes you make to `utils.py` in Colab to a local copy. Each time you re-start a session the changes of any file in the files folder are lost.

**Grading:**

- This homework is worth 10 points.
- Each function you code in `utils.py` is worth 1 point, for a total of 7 points.
- Each answer in part 5 is worth 1 point, for a total of 3 points.
- Points will be automatically assigned when passing tests, and manually assigned when it comes to your written responses.

**Academic integrity:**

Remember that this homework should be authored by you only. It's ok to discuss with classmates but you have to submit your own original solution.

-------------------------------------------------------------

## Sound Classification for Instrument Recognition

In this homework, we will explore the task of recognizing musical instruments based solely on their sound profiles. This involves distinguishing the nuanced tonal characteristics that each instrument produces, such as the sharp resonance of a violin versus the deep hum of a cello or the distinct timbre of a flute compared to a clarinet.

Starting from an available dataset of samples from different instruments, we will extract timbre-related features from the audio. Once processed, the data will be segmented into training, validation, and testing sets to ensure the model's robustness and generalization capabilities.

Using a simple model, we will analyze and critic its performance, trying to explain its behaviour to be able to improve it in the future.
