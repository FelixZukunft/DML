"""Visualise the distribution of Curricular units 2nd sem (grade)."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RAW = Path(__file__).parents[1] / "data" / "raw" / "students_raw.csv"
OUT = Path(__file__).parents[1] / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(RAW)
col = "Curricular units 2nd sem (grade)"
grades = df[col]

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(grades, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)

mean, median = grades.mean(), grades.median()
ax.axvline(mean,   color="#DD4444", linewidth=1.5, linestyle="--", label=f"Mean {mean:.2f}")
ax.axvline(median, color="#44AA66", linewidth=1.5, linestyle="-",  label=f"Median {median:.2f}")

ax.set_title("Curricular units 2nd sem (grade) — outcome distribution", fontsize=13, pad=12)
ax.set_xlabel("Grade (0 = withdrew / failed all units)", fontsize=11)
ax.set_ylabel("Students", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
out_path = OUT / "grade_distribution.png"
fig.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
plt.show()
