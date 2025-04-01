<h1 align="center">📊 Student Performance Analyzer 📊</h1>

<p align="center">
  <strong>A comprehensive, client-side web application for instant analysis and visualization of student performance data.</strong>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#️-how-to-use">How To Use</a> •
  <a href="#-technology-stack">Tech Stack</a> •
  <a href="#-statistical-terms-explained">Stats Explained</a> •
  <a href="#-contributing">Contributing</a> •
  <a href="#-license">License</a>
</p>

<p align="center">
  <a href="https://github.com/your-username/your-repo-name"> <!-- Optional: Replace repo link -->
    <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status: Active">
  </a>
  <a href="LICENSE"> <!-- Optional: Add a LICENSE file and link -->
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License: MIT">
  </a>
</p>

---

## ✨ Key Features

This tool empowers educators and students with detailed performance insights:

*   ✅ **Comprehensive Statistical Analysis:**
    *   **Individual Metrics:** Score, Rank, Percentile, Z-Score, Comparison vs. Mean/Median/Target.
    *   **Class Metrics:** Mean, Median, Mode(s), Standard Deviation, Variance, Min/Max, Range, Quartiles (Q1, Q3), IQR, Skewness, Kurtosis.
*   ✅ **Interactive Visualization:**
    *   Dynamic Histogram & Frequency Polygon (Chart.js).
    *   Highlights the student's score bin.
    *   Download chart as PNG.
*   ✅ **Frequency Distribution Table:**
    *   Clear breakdown of score counts per histogram bin.
*   ✅ **Percentile Lookup Tool:**
    *   Interactive slider to map scores to percentile ranks.
*   ✅ **Modern & User-Friendly Interface:**
    *   Clean, responsive design (Bootstrap 5).
    *   Elegant dark theme included.
*   ✅ **Flexible Input:**
    *   Accepts scores via commas, spaces, or new lines.
    *   Robustly ignores non-numeric entries.
*   ✅ **Instant Feedback:**
    *   Live count of valid scores detected.
    *   Clear validation messages for input errors.
*   ✅ **Convenience Tools:**
    *   "Load Example" button for quick demo.
    *   "Copy Stats" button with clipboard fallback.
*   ✅ **Privacy Focused:**
    *   **100% Client-Side:** All calculations happen in *your* browser. No data is ever sent to a server.
*   ✅ **Accessibility:**
    *   Semantic HTML and ARIA attributes implemented.

---

## 🛠️ How To Use

Get started in two simple steps:

1.  **Download:** Obtain the `index.html` file containing the application code.
2.  **Open:** Launch the downloaded `index.html` file directly in your modern web browser (e.g., Chrome, Firefox, Edge, Safari).

**Using the Analyzer Interface:**

1.  **Enter Your Score:** Input the individual score (required).
2.  **Enter Target Score (Optional):** Add a passing/target score for comparison.
3.  **Enter Class Scores:** Paste or type all class scores, separated by commas, spaces, or new lines.
4.  **Analyze:** Click the **`Analyze Performance`** button.
5.  **View Results:** Scroll down to see the detailed analysis, charts, and tables.
6.  **Explore:** Use the percentile slider, download the chart, or copy stats.
7.  **(Optional) Load Example:** Click **`Load Example`** to auto-fill sample data.
8.  **Clear:** Click **`Clear All`** to reset everything.

---

## 💻 Technology Stack

This application is built using modern web technologies:

*   **Core:** HTML5, CSS3 (with CSS Variables)
*   **Framework:** [Bootstrap 5.3](https://getbootstrap.com/) - *For responsive layout & UI components.*
*   **Charting:** [Chart.js 4.4](https://www.chartjs.org/) - *For interactive data visualization.*
*   **Statistics:** [simple-statistics 7.8](https://simplestatistics.org/) - *For client-side statistical calculations.*
*   **Icons:** [Bootstrap Icons 1.11](https://icons.getbootstrap.com/) - *For clear iconography.*
*   **Logic:** Vanilla JavaScript (ES6+)

---

## 🧠 Statistical Terms Explained

> Understanding the numbers enhances the analysis. Here's a quick reference for the key terms used:
>
> *   **Mean (Average):** Sum of scores / Number of scores. Affected by extreme values.
> *   **Median (Q2):** The middle score when data is sorted. Less affected by extremes.
> *   **Mode:** The most frequent score(s).
> *   **Standard Deviation (Std. Dev):** Average distance of scores from the mean. Measures spread. (Uses Sample Std Dev).
> *   **Variance:** The square of the Standard Deviation. (Uses Sample Variance).
> *   **Range:** Difference between Highest and Lowest scores.
> *   **Quartiles (Q1, Q3):** Points dividing the sorted data into four equal parts. Q1=25th percentile, Q3=75th percentile.
> *   **IQR (Interquartile Range):** The range of the middle 50% of scores (Q3 - Q1). Robust measure of spread.
> *   **Percentile:** Percentage of scores *strictly lower* than a specific score.
> *   **Rank:** Position relative to others (1st = highest).
> *   **Z-Score:** How many standard deviations a score is from the mean. Measures how typical a score is.
> *   **Skewness:** Measures distribution asymmetry (Left/Negative vs. Right/Positive).
> *   **Kurtosis (Excess):** Measures "tailedness" / "peakiness" vs. a normal distribution (Peaked/Leptokurtic vs. Flat/Platykurtic).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1.  **Fork** the repository (if hosted on GitHub/GitLab).
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the Branch (`git push origin feature/AmazingFeature`).
5.  Open a **Pull Request**.

Alternatively, open an [Issue](https://github.com/your-username/your-repo-name/issues) <!-- Optional: Replace with actual Issues link --> with the tag "enhancement" or "bug".

---

## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for full details. <!-- Optional: Create a LICENSE file and ensure it exists -->

---

## 🙏 Acknowledgements

Special thanks to the creators and maintainers of these fantastic open-source libraries:

*   [Bootstrap](https://getbootstrap.com/)
*   [Chart.js](https://www.chartjs.org/)
*   [simple-statistics](https://simplestatistics.org/)
*   [Bootstrap Icons](https://icons.getbootstrap.com/)

---
