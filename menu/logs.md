---
layout: page
title: Logs - Takeaways From Others
---
<ul class="posts">
  {% for post in site.categories.logs %}

    <li itemscope>
      <a href="{{post.logurl}}">"{{ post.title }}" - by {{ post.author }}</a>
      <div class="post-excerpt">{{post.excerpt}}
      </div>
    </li>

  {% endfor %}
</ul>
