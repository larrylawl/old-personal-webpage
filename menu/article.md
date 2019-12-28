---
layout: page
title: Articles
---
<ul class="posts">
  {% for article in site.categories.article %}

    {% unless article.next %}
      <h3>{{ article.date | date: '%Y' }}</h3>
    {% else %}
      {% capture year %}{{ article.date | date: '%Y' }}{% endcapture %}
      {% capture nyear %}{{ article.next.date | date: '%Y' }}{% endcapture %}
      {% if year != nyear %}
        <h3>{{ article.date | date: '%Y' }}</h3>
      {% endif %}
    {% endunless %}

    <li itemscope>
      <a href="{{ site.github.url }}{{ article.url }}">{{ article.title }}</a>
      <p class="post-date"><span><i class="fa fa-calendar" aria-hidden="true"></i> {{ article.date | date: "%B %-d" }} - <i class="fa fa-clock-o" aria-hidden="true"></i> {% include read-time.html %}</span></p>
    </li>

  {% endfor %}
</ul>
