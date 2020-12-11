---
layout: post
title: "Unix Notes"
author: "Larry Law"
categories: notes
hidden: true
---

# Unix Notes
## Server Commands
`nohup`: Stands for no hang up. Nohup is very helpful when you have to execute a shell-script or command that take a long time to finish.
`# nohup command-with-options &`

`nice`: run a program with modified scheduling priority
`nice [OPTION] [COMMAND [ARG]...]`

`screen`: start a screen session and then open any number of windows (virtual terminals) inside that session. Processes running in Screen will continue to run when their window is not visible even if you get disconnected.
`screen`

## Pipe Commands
*Pipe is a form.*

`join`: joining lines of two files on a common field. 
`$join [OPTION] FILE1 FILE2`

`paste`: join files horizontally by outputting lines consisting of lines from each file specified, separated by tab as delimiter, to the standard output.
`paste [OPTION]... [FILES]...`

> What's the difference between `join` and `paste`?

The join command takes two files and merges their columns—as long as both files share a common field. In other words, for join to work properly, there must be a common field for each row of data (such as row number). The common field functionality will keep a user from merging incorrect data together.

`cut`: cutting out the sections from each line of files and writing the result to standard output.
`cut OPTION... [FILE]...`

`sort`: arranging the records in a particular order. By default, the sort command sorts file assuming the contents are ASCII.

## Unix Piping
`|`: data flows from left to right through the pipeline.
`command_1 | command_2 | command_3 | .... | command_N `

`>`: Redirect output to a file
`ls > list.txt`

`2>`: Redirect `stderr` (file descriptor for `stderr` is 2).

`2>&1`: Redirect the stderr to the same place we are redirecting the stdout

`tee`: tee command reads the standard input and writes it to both the standard output and one or more files
`echo 'foo' | tee foo.txt`