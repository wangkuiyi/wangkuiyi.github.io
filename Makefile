%.html: %.md head1.html
	pandoc $< -o $@ --mathjax
	cp head1.html temp.html
	cat $@ >> temp.html
	mv temp.html $@

%.svg: %.dot
	dot -Tsvg $< -o $@

all: $(patsubst %.dot,%.svg,$(wildcard *.dot)) $(patsubst %.md,%.html,$(wildcard *.md))
