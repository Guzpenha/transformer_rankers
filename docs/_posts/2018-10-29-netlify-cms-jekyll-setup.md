---
type: post
title: SparrowX Setup for Netlify CMS
seotitle: How to Add Netlify CMS to A Jekyll Site - SparrowX Documentation
author: Mighil
description: Learn how to add Netlify CMS to a Jekyll site. This tutorial is based on SparrowX theme.
url: /netlify-cms-jekyll-setup/
tags:
  - documentation
---

## Getting Started

* Create a Netlify account if you don't have one.
* [Fork SparrowX](https://github.com/mighildotcom/sparrowx/). (No need enable GitHub pages since we'll be using Netlify CMS to fetch, build the repo and point the domain.)
* Visit https://app.netlify.com/ and click **"New site from Git"**

## Netlify deploy settings for SparrowX

* Select ```master branch``` to deploy. Use ```jekyll build``` build command and set the Publish directory as ```_site```.

![Netlify deploy settings for Jekyll](/images/netlify-jekyll-deploy-settings.png)

## /admin/ directory explained

The **/admin/** directory contains the ```index.html``` and ```config.yml``` for Netlify CMS. 

Here's how the ```config.yml``` looks for now.

```
backend:
  name: git-gateway
  branch: master

publish_mode: editorial_workflow

media_folder: "images" # Media files will be stored in the repo under images
public_folder: "/images" # The src attribute for uploaded media will begin with /images

collections:
  - name: "post"
    label: "Post"
    folder: "_posts"
    create: true
    slug: "{{year}}-{{month}}-{{day}}-{{slug}}"
    fields:
      - {label: "Type", name: "type", widget: "hidden", default: "post"}
      - {label: "Title", name: "title", widget: "string"}
      - {label: "SEO Title", name: "seotitle", widget: "string"}
      - {label: "Author", name: "author", widget: "string"}
      - {label: "Description", name: "description", widget: "string", required: false}
      - {label: "OG Image", name: "headerimage", widget: "string", required: false}
      - {label: "Updated Date", name: "updated", widget: "string", required: false}
      - {label: "Body", name: "body", widget: "markdown", required: false}
      - {label: "Tags", name: "tags", widget: "string"}
      - {label: "URL", name: "url", widget: "string"}

  - name: "page"
    label: "Page"
    folder: "_pages"
    create: true
    slug: ".md"
    fields:
      - {label: "Type", name: "type", widget: "hidden", default: "page"}
      - {label: "Title", name: "title", widget: "string"}
      - {label: "SEO Title", name: "seotitle", widget: "string"}
      - {label: "Description", name: "description", widget: "string", required: false}
      - {label: "URL", name: "url", widget: "string"}
      - {label: "Body", name: "body", widget: "markdown", required: false}
```

You may edit this according to your preference. 

## Authentication

Credit: [This blog post](https://www.chrisanthropic.com/blog/2018/adding-netlifycms-to-jekyll-secure-with-netlify-identity-git-gateway/)

You've to perform few tweaks within your Netlify account to connect Netlify and GitHub.

Netlify offers an Identity service that allows you to manage authenticated users with only an email or optional SSO with GitHub, Google, GitLab, and BitBucket.

### Enable Netlify Identity

* Log in to Netlify, Domain Settings » Identity » Enable Identity Service
* Leave registration open for now
* Tick the box of any External Providers you want to support
* Leave all the other defaults as-is for now

### Enable Git Gateway

This step obviously requires a GitHub account.

* Scroll to the bottom of the page
* Click on “Enable Git Gateway”
* Log in to your GitHub account when prompted

### Add The Netlify Identity Widget

Now we need to add a small “Identity Widget” script to the admin page and main page of the site. Rather than modify my Jekyll template I’m going to hightlight another cool feature Netlify provides - Script Injection. In short we can tell Netlify to inject JavaScript snippets into the </head> or before the end of the </body> tags on every page of the site.

**Log in to Netlify, Site Settings » Build & Deploy » Post Processing » Snippet Injection**

* Insert before: select </head>
* Script name: Netlify Identity Widget
* HTML: 

```
<script src="https://identity.netlify.com/v1/netlify-identity-widget.js"></script>
```

### Add The Netlify Identity Widget Redirect

Now that we have code to handle the login, let’s make sure we get redirected to the amin page after logging in.

**Log in to Netlify, Site Settings » Build & Deploy » Post Processing » Snippet Injection**

* Insert before: select </head>
* Script name: Netlify Identity Redirect
* HTML:

```
<script>
  if (window.netlifyIdentity) {
    window.netlifyIdentity.on("init", user => {
      if (!user) {
        window.netlifyIdentity.on("login", () => {
          document.location.href = "/admin/";
        });
      }
    });
  }
</script> 
```

Now Netlify will inject that code into every page on your site.

### Test it out

Push your changes and watch Netlify rebuild your site with a new admin dashboard!

### Approve Account Creation

You should have recieved an email by now from Netlify (at whatever email you use for your Netlify account) that you were added as a user to the site and that you need to click the link to confirm. This is the Netlify Identity service at work! It automatically invites you since you’re the Netlify admin.


* Open the Netlify email and click “Accept Invitation”
* Create a password when prompted
* Turn off open registration! (Now that we’ve successfully registered there’s no need to keep it open)

**Log in to Netlify, Domain Settings » Identity » Registration Preferences » Invite Only**

### Login Via Email

Visit ```YOURDOMAIN/admin``` and login via email (username/password) first to test that it’s working. Then test via Google/GitHub/etc.

### The Netlify CMS Dashboard

The dashboard looks the way to configured on ```config.yml``` inside ```/admin```.

Here's the basics:

![Netlify CMS dashboard](/images/netlify-cms-jekyll-theme.png)

### How tags work

You should create specific .md files within ```/tag/``` before using the variable.
