open -a Preview \
  $(ls -1 ../figures/A* | sort -r | head -n 1) \
  $(ls -1 ../figures/B* | sort -r | head -n 1) \
  $(ls -1 ../figures/C* | sort -r | head -n 1) \
  $(ls -1 ../figures/D* | sort -r | head -n 1) \
  $(ls -1 ../figures/E* | sort -r | head -n 1)

