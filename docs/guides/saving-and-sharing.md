# Saving and sharing biplots

`plot()` returns a {class}`~bipl5.render.widget.Bipl5Widget` with three
ways out:

## In notebooks

Evaluate the widget and it displays itself (Jupyter, JupyterLab, VS
Code, Quarto). The embedded page loads plotly.js from the CDN, so the
notebook file stays small.

```python
w = bp.plot()
w          # renders inline
```

## Standalone HTML files

```python
w.save("biplot.html")
```

writes a **fully offline** file: plotly.js and the bipl5 JavaScript are
embedded (about 5 MB), so the file works with no internet connection —
attach it to an email, drop it on a share, or open it years later.

To produce a smaller file that fetches plotly.js from the CDN instead:

```python
path = w.save("biplot.html", include_plotlyjs="cdn")
```

## Quick preview

```python
w.show()   # writes a temp file and opens the default browser
```

## Fine control

`w.to_html(full_html=..., include_plotlyjs=..., div_id=...)` returns the
HTML string, and `w.figure` / `w.payload` expose the underlying plotly
figure dict and the JavaScript payload for inspection or customization.

```{note}
bipl5 pins **plotly.js 2.35.2** — the interactive layer is shared with
the R package, which targets the plotly.js 2.x API. `include_plotlyjs`
accepts `True` (embed the vendored copy), `"cdn"` (pinned CDN URL), or
any explicit `.js` URL.
```
