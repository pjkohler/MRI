#!/bin/bash
set -e

PROGNAME=$(basename $0)

#die() {
#    echo "$PROGNAME: $*" >&2
#    exit 1
#}

usage() {
    if [ "$*" != "" ] ; then
        echo "Error: $*"
    fi

    cat << EOF
Usage: $PROGNAME [OPTION ...] [foo] [bar]
<Program description>.

Options:
  -h, --help          display this usage message and exit
  -d, --delete        delete things
  -o, --output [FILE] write output to file
EOF

    exit 1
}

foo=""
bar=""
delete=0
output="-"
args=()
while [ $# -gt 0 ] ; do
    case "$1" in
    -h|--help)
        usage
        ;;
    -d|--delete)
        delete=1
        ;;
    -o|--output)
        output="$2"
        shift
        ;;
    -*)
        usage "Unknown option '$1'"
        ;;
    *)
        args=("${args[@]}" "$1")
        ;;
    esac
    shift
done

cat <<EOF
foo=$foo
bar=$bar
delete=$delete
output=$output
EOF

for arg in "${args[@]}" ; do
    echo "'$arg'"
done