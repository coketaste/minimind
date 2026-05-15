MAIN = main

.PHONY: all clean view

all:
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

view: all
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || true

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.pdf
