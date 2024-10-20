%.html: %.md
	pandoc $< -o $@ --mathjax
	echo '<script src="https://cdn.plot.ly/plotly-2.8.3.min.js"></script>' > temp.html
	echo '<link rel="stylesheet" href="https://jingnanshi.com/static/main.css" />' >> temp.html
	echo '<link rel="stylesheet" href="https://jingnanshi.com/static/code.css" />' >> temp.html
	echo '<script src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.1/highlight.min.js"></script>' >> temp.html
	echo '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>' >> temp.html
	cat $@ >> temp.html
	mv temp.html $@

%.svg: %.dot
	dot -Tsvg $< -o $@

all: $(patsubst %.dot,%.svg,$(wildcard *.dot)) $(patsubst %.md,%.html,$(wildcard *.md))
