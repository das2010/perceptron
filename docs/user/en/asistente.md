# The guided assistant

The simplest way to use Perceptron: describe the problem in your own words and the assistant takes
you step by step to a validated model. No machine-learning knowledge needed.

## Step by step

1. **Describe the problem.** On **Home**, write what you want to achieve, for example “I have a
   sales spreadsheet and want to estimate how much we will sell”. Click **Start**.
2. **Answer the questions.** The assistant asks one thing at a time, with examples: what data you
   have, what you want to get, which error is worse, what result is good enough. Answer by typing
   or with the control below. In **What I understood** you can correct anything. When everything
   is there, review it and click **Yes, that's it**.
3. **Prepare the data.** The assistant shows how it must look, with an example table (or how to
   arrange the ZIP folders), a template to download and tips. Upload the file.
4. **Review.** Perceptron checks the data and tells you what it found. If something blocks
   progress (for example, too few examples), it tells you what to fix.
5. **Training.** Click **Prepare and train**. Perceptron designs the model, trains it and
   evaluates it on examples it never saw. You see the current stage and can leave: it notifies you
   when done.
6. **Result.** It explains how often it is right, or how far off it is on average, and whether it
   meets your goal. If it found an exact formula, you can copy it for Excel or Python.
7. **Validate with new cases.** Enter cases that are not in the data you uploaded, with their real
   result. The model predicts without seeing it and then they are compared. For tables you can
   upload a CSV with several cases. With at least 10 cases (and every category) it is ready to use.
8. **Use it.** Fill in only the data the model needs, or upload the photo or audio, and get the
   prediction.

## Which cases it solves

- **Spreadsheets:** pick one of several answers, estimate a number, discover a formula, forecast
  the next values of something you measure over time (sales, consumption, wear) or spot unusual
  moments in measurements (even without marked cases).
- **Photos:** say what kind each one is, locate where the objects are, mark the exact area (for
  example the damaged part) or read a text (a label, a serial number).
- **Texts and recordings:** pick one of several answers.

If your case does not fit as is, the assistant suggests an equivalent way to frame it. For
example, grouping into ranges a number you want to estimate from photos.

## Advanced options

All technical tools (data, design, training, experiments, models, monitoring) remain under
**Advanced options** inside the project. See [Your first model](primer-modelo.md) for the manual
walkthrough.
