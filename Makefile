roofline.html: roofline.md
	pandoc roofline.md -o roofline.html --mathjax
	echo '<link rel="stylesheet" href="https://cdn.simplecss.org/simple.css">' > temp.html
	echo '<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>' >> temp.html
	cat roofline.html >> temp.html
	mv temp.html roofline.html
