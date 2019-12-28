---
layout: post
title: "Global, nonlocal, and local values in Python"
author: "Larry Law"
categories: articles
tags: [python]
image: python.jpg
---

## The problem

Why am I getting an `UnboundLocalError` when my variable has a value?

```py
x = 10
def foo():
    print(x)
    x += 1

foo()
```

(Skip to the documentation directly [here](https://docs.python.org/3/faq/programming.html).
Read on if you don't understand the docs)

## The explanation

This explanation from the docs clears things up:

> In Python, variables that are only referenced inside a function are implicitly global. <br />

> If a variable is assigned a value anywhere within the function’s body, it’s assumed to be a local unless explicitly
> declared as global.

Line 2 is implemented so as to prevent programmers from unintentionally manipulating the value of global variables.

## The solution

Use the `global` keyword. This explicit declaration tells python (and yourself) that you are referencing the
`global` variable.

```py
x = 10
def foo():
    global x
    print(x)
    x += 1

foo()
```

Use the `nonlocal` keyword. `nonlocal` causes the listed identifier to refer to _previously bound variables_ in
the _nearest enclosing scope excluding globals_. Note that `for-loops` in Python do not create a local scope (unlike Java).
Docs [here](https://docs.python.org/3/reference/simple_stmts.html#grammar-token-nonlocal-stmt)

```py
def foo():
    x = 10
    def bar():
        nonlocal x
        print(x)
        x += 1
    bar()
    print(x)
foo()
```
