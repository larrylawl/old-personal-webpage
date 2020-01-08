---
layout: page
title: Articles
---
<ul class="posts">
  <div class ="tags">
    <b>Tags:</b>
    <ul>
      {% for tag in site.tags %}
        {% assign t = tag | first %}
        {% assign posts = tag | last %}
        <li>
        <a href="#{{t}}"> {{t}} ({{ posts | size }}) </a>
        </li>
        
      {% endfor %}
    </ul> 
  </div>

  <div class ="articles">
    {% for tag in site.tags %}
      {% assign t = tag | first %}
      {% assign posts = tag | last %}

      <h2><a id="{{t}}">{{t}}</a></h2>

      {% for post in posts %}
        {% if post.tags contains t %}
        <li itemscope>
          <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
          <p class="post-date"><span><i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%B %-d, %Y" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
        </li>
        {% endif %}
      {% endfor %}

    {% endfor %}
  </div>
</ul>
