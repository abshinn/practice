#!/usr/bin/env python -tt -B

import applestock

def main():
    print applestock.profit([4.5, 2.0, 3.3, 4.75, 2.75])
    print applestock.profit([4, 2, 3, 5, 2])
    print applestock.profit([5, 2, 3, 3, 2])

if __name__ == "__main__":
    main()
