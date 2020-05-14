---
layout: page
title: Takeaways From Others
---
<ul class="posts">
  <div class ="articles">
    {% for post in site.categories.takeaways %}
      <li itemscope>
        <a href="{{ site.github.url }}{{ post.url }}">{{ post.title }}</a>
        <p class="post-date"><span><i class="fa fa-calendar" aria-hidden="true"></i> {{ post.date | date: "%B %-d, %Y" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
      </li>
    {% endfor %}
  </div>
</ul>
