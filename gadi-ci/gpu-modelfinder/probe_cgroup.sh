#!/bin/bash
#PBS -N cgprobe
#PBS -P dx61
#PBS -l storage=scratch/dx61+scratch/rc29
#PBS -l wd
#PBS -j oe
echo "==== cgroup probe $(hostname) $(date -Iseconds) ===="
echo "--- /proc/self/cgroup ---"; cat /proc/self/cgroup
echo "--- physical MemTotal ---"; grep MemTotal /proc/meminfo
echo "--- v2 unified path memory.max ---"
v2path=$(awk -F: '/^0::/{print $3}' /proc/self/cgroup)
[ -n "$v2path" ] && cat "/sys/fs/cgroup${v2path}/memory.max" 2>/dev/null && echo "(v2 above)"
echo "--- v1 memory path limit_in_bytes ---"
v1path=$(awk -F: '/:memory:/{print $3}' /proc/self/cgroup)
echo "v1path=$v1path"
cat "/sys/fs/cgroup/memory${v1path}/memory.limit_in_bytes" 2>/dev/null && echo "(v1 scoped above)"
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null && echo "(v1 root above)"
echo "--- search for any 180GB-ish limit ---"
find /sys/fs/cgroup/memory -name memory.limit_in_bytes 2>/dev/null | while read f; do v=$(cat "$f" 2>/dev/null); [ "$v" -lt 1000000000000 ] 2>/dev/null && echo "$f = $v"; done | head
echo "==== done ===="
