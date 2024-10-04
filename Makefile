%.html: %.md
	pandoc $< -o $@ --mathjax
	echo '<link rel="stylesheet" href="https://cdn.simplecss.org/simple.css">' > temp.html
	echo '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>' >> temp.html
	cat $@ >> temp.html
	mv temp.html $@

%.svg: %.dot
	dot -Tsvg $< -o $@

all: $(patsubst %.dot,%.svg,$(wildcard *.dot)) $(patsubst %.md,%.html,$(wildcard *.md))
