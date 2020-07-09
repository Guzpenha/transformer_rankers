---
title: How To Use This Theme
type: post
seotitle: SparroX - Best Minimal Jeykll Theme Documentation and Download
author: Mighil
description: Read the documenation for SparrowX minimal and clean Jekll theme. 
url: /documentation/
tags:
  - documentation
---

## Documentation

Here's the basic documentation for SparrowX. [Visit this page](/netlify-cms-jekyll-setup) to read the documentation for Netlify CMS.

## Install

Just [fork this repo](https://github.com/mighildotcom/sparrowx/fork) and replace the posts with your own. Also rename the sample poetry collection to replace it with your own collection, or just delete it if you don't need to use collections. The example is poetry, but you can easily revamp this into essays, short stories, portfolio, or whatever your creative work is. 

### General configuration

Customize these values in the `_config.yml` file: 

```yaml
title                    : "SparrowX"
description              : "A clean minimal Jekyll theme for SEO-focused writers."
favicon                  : "/assets/images/image.png" # You can use base64 encoded images.
url                      : "https://sparrowx.mighil.com" # the base hostname & protocol for your site e.g. "https://mmistakes.github.io"
baseurl                  : "" # the subpath of your site, e.g. "/blog" leave blank if no subpath

# Site Author
author:
  name                   : "Mighil" # appears in the footer and atom feed
```


### Disqus

To activate Disqus, fill in your disqus shortname: 

```yaml
comments                 : true # set it to false to disable comments
disqus:
  shortname              : your_shortname
```

### Google analytics

Enable google analytics by filling in your tracking ID: 

```yaml
analytics:
  google:
    tracking_id          : # add your google tracking id here
```

### Collections

If you are using collections, be sure to replace this with your collection name: 

```yaml
collections:
  poetry:
    output: true
    permalink: /:collection/:title/
```

### Pagination

Currently, pagination is set to a the blog page. Unfortunately Jekyll does not yet support pagination on multiple pages. 

If you want to paginate on a different page, simply change the `paginate_path` value in the config file and create the relevant page. 

```yaml
paginate: 5 # amount of posts to show
paginate_path: "/blog/page:num/"
timezone: # http://en.wikipedia.org/wiki/List_of_tz_database_time_zones
```

### Navigation

You can change the navigation links in `_data/navigation.yml`.

### Front matter defaults

At the bottom of the config file, you can also set [front matter defaults](https://jekyllrb.com/docs/configuration/#front-matter-defaults) so that you don't have to repeat the same front matter for every post. This is an example: 

```yaml
# Front matter defaults
defaults:
  # _pages
  - scope:
      path: ""
      type: pages
    values:
      layout: page
      comments: false
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: post
      comments: true
      share: true
  # _poetry
  - scope:
      path: ""
      type: poetry
    values:
      layout: post
      share: false
```

### How tags work

You should create specific .md files within ```/tag/``` before using the variable.

### Other

To enhance SEO, you can add your twitter username to `_config.yml`. 

You can also add an image path for the `og_image` value in the config file, for a default open graph sharing image. This value can be overridden in individual posts by using **headerimage** variable. 

Your default image Netlify CMS uploads will go to /images/.

For the various config options see the [sample config file for the demo site](https://github.com/mighildotcom/sparrowx/blob/master/_config.yml)

The color schemes and fonts can also be customized through scss, by editing the relevant variable in `_variables.scss`. 
