#!/usr/bin/env bash
# Reprint a text file every time it changes. Ctrl+C to stop.
#   bash watch/watch_file.sh                 # watches watch/file.txt
#   bash watch/watch_file.sh path/to/other.txt
#   touch watch/file.txt                     # force a redraw / status refresh
#
# When a line is itself a path to a log, it is annotated with the log's state:
# a log holding "------ FINISHED ------" is done, anything else is still going.
f="${1:-watch/file.txt}"
touch "$f"

status() {  # $1 = a log path
  [[ -f $1 ]] || { echo "-- missing"; return; }
  grep -q -- "------ FINISHED ------" "$1" && echo "-- finished" || echo "-- in progress"
}

# Redraw when the STATUSES change, not when the logs do. The logs are rewritten
# every few seconds, so following their timestamps made the terminal blink
# constantly; following the statuses means one redraw when a run finishes.
sig() {
  stat -c %Y.%s "$f" 2>/dev/null
  while IFS= read -r line || [[ -n $line ]]; do
    [[ $line == /* ]] && status "$line"
  done < "$f"
}

last=""
while true; do
  now=$(sig)
  if [[ "$now" != "$last" ]]; then
    clear
    echo "── $f  ($(date +%H:%M:%S)) ──"
    while IFS= read -r line || [[ -n $line ]]; do
      if [[ -n $line && $line != \#* && $line == /* ]]; then
        printf '%s  %s\n' "$line" "$(status "$line")"
      else
        printf '%s\n' "$line"
      fi
    done < "$f"
    last="$now"
  fi
  sleep 0.5
done
