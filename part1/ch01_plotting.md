# 1.2 What's Happening? Plotting Data

The very simplest way to present or visualize a dataset is to produce a table. Tables can be helpful, but aren't much use for large datasets, because it is difficult to get any sense of what the data means from a table. We need more effective visualization tools.

## 1.2.1 Bar Charts

A **bar chart** is a set of bars, one per category, where the height of each bar is proportional to the number of items in that category. A glance at a bar chart often exposes important structure in data, for example, which categories are common, and which are rare.

Bar charts are particularly useful for **categorical data**. For example, consider the Chase and Danner student dataset that records:
- Gender of students (Boy/Girl)
- Goals students value (Grades/Popular/Sports)

```{figure} images/fig_1_1_bar_charts.png
:name: fig-bar-charts
:width: 90%

Bar charts from the Chase and Danner study. Left: Number of children of each gender - notice the bars are about the same height, indicating roughly equal numbers of boys and girls. Right: Number of children selecting each of three goals - you can tell at a glance that different goals are more or less popular by looking at the height of the bars.
```

### Key Properties of Bar Charts

- **One bar per category**: Each distinct category gets its own bar
- **Height represents count**: The height of each bar is proportional to the number of data items in that category
- **Visual comparison**: You can compare categories at a glance by comparing bar heights
- **Best for categorical data**: Bar charts work best when your data falls into discrete categories

## 1.2.2 Histograms

Data is **continuous** when a data item could take any value in some range or set of ranges. In turn, this means that we can reasonably expect a continuous dataset contains few or no pairs of items that have exactly the same value. 

Drawing a bar chart in the obvious way—one bar per value—produces a mess of unit height bars, and seldom leads to a good plot. Instead, we would like to have fewer bars, each representing more data items. We need a procedure to decide which data items count in which bar.

A simple generalization of a bar chart is a **histogram**. We divide the range of the data into intervals, which do not need to be equal in length. We think of each interval as having an associated "pigeonhole," and choose one pigeonhole for each data item. We then build a set of boxes, one per interval. Each box sits on its interval on the horizontal axis, and its height is determined by the number of data items in the corresponding pigeonhole.

```{figure} images/fig_1_2_histograms.png
:name: fig-histograms
:width: 90%

Histograms for continuous data. Left: Net worths for 10 individuals - there are five bars, and the height of each bar gives the number of data items that fall into its interval. The picture suggests that net worths tend to be quite similar, and around $100,000. Right: Cheese goodness scores for 20 cheeses - there are six bars (0–10, 10–20, and so on). You can see at a glance that quite a lot of cheeses have relatively low scores, and few have high scores.
```

### Understanding Histograms

- **Intervals (bins)**: The horizontal axis is divided into intervals
- **Counts**: The height of each box represents the number of data items in that interval
- **Reveals distribution**: Histograms show how data is distributed across a range of values
- **Pattern detection**: You can spot clustering, gaps, and outliers

## 1.2.3 How to Make Histograms

Usually, one makes a histogram by finding the appropriate command or routine in your programming environment. It is useful to understand the procedures used to make and plot histograms.

### Histograms with Even Intervals

The easiest histogram to build uses equally sized intervals. Write $x_i$ for the $i$-th number in the dataset, $x_{\min}$ for the smallest value, and $x_{\max}$ for the largest value. We divide the range between the smallest and largest values into $n$ intervals of even width $(x_{\max} - x_{\min})/n$.

In this case, the height of each box is given by the number of items in that interval. We could represent the histogram with an $n$-dimensional vector of counts. Each entry represents the count of the number of data items that lie in that interval.

**Important**: We need to be careful to ensure that each point in the range of values is claimed by exactly one interval. For example:
- We could have intervals of $[0, 1)$ and $[1, 2)$
- We could have intervals of $(0, 1]$ and $(1, 2]$
- We could NOT have intervals of $[0, 1]$ and $[1, 2]$ (the value 1 would appear in two boxes)
- We could NOT have intervals of $(0, 1)$ and $(1, 2)$ (the value 1 would not appear in any box)

### Histograms with Uneven Intervals

For a histogram with even intervals, it is natural that the height of each box is the number of data items in that box. But a histogram with even intervals can have empty boxes. In this case, it can be more informative to have some larger intervals to ensure that each interval has some data items in it.

**Rule for uneven intervals**: Plot boxes such that **the area of the box is proportional to the number of elements in the box**.

Write:
- $dx$ for the width of the intervals
- $n_1$ for the height of the box over the first interval (the number of elements in the first box)
- $n_2$ for the height of the box over the second interval

If we fuse two consecutive intervals:
- The height of the fused box should be $(n_1 + n_2)/2$
- The area of the first box is $n_1 \cdot dx$
- The area of the second box is $n_2 \cdot dx$
- The area of the fused box is $(n_1 + n_2) \cdot dx$

For each of these boxes, the area is proportional to the number of elements.

## 1.2.4 Conditional Histograms

Most people believe that normal body temperature is 98.4°F. If you take other people's temperatures often (for example, you might have children), you know that some individuals tend to run a little warmer or a little cooler than this number.

Consider a dataset giving the body temperature of a set of individuals. As you can see from the histogram, the body temperatures cluster around a small set of numbers. But what causes the variation?

### Investigating with Conditional Histograms

One possibility is gender. We can investigate this possibility by comparing:
- A histogram of temperatures for males
- A histogram of temperatures for females

```{figure} images/fig_1_3_conditional_histograms.png
:name: fig-conditional-histograms
:width: 80%

Conditional histograms for body temperature data. Top: Histogram of all body temperatures, showing clustering around one value (marked with red dashed line at 98.4°F). Middle and Bottom: Histograms for each gender separately. It looks as though one gender runs slightly cooler than the other, suggesting that gender may influence body temperature.
```

Histograms that plot only part of a dataset are sometimes called **conditional histograms** or **class-conditional histograms**, because each histogram is conditioned on something. In this case, each histogram uses only data that comes from a particular gender.

The conditional histograms suggest that individuals of one gender run a little cooler than individuals of the other. Being certain takes considerably more work than looking at these histograms, because the difference might be caused by an unlucky choice of subjects. But the histograms suggest that this work might be worth doing.

### When to Use Conditional Histograms

- **Compare subgroups**: When you want to see if different groups behave differently
- **Identify patterns**: To understand what factors contribute to variation in your data
- **Hypothesis generation**: Conditional histograms can suggest relationships worth investigating further

```{admonition} Key Takeaway
:class: tip
Conditional histograms help you investigate whether a categorical variable (like gender, age group, or treatment type) affects the distribution of a continuous variable (like temperature, height, or test scores).
```

## Summary

In this section, we learned about:

1. **Bar charts**: For visualizing categorical data
2. **Histograms**: For visualizing continuous data distributions
3. **Making histograms**: Both with even and uneven intervals
4. **Conditional histograms**: For comparing distributions across different subgroups

These plotting techniques are your first tools for answering the question "What's going on here?" with data. Always start by visualizing your data before computing summaries or performing statistical tests.
