# coding: utf-8
 
Gem::Specification.new do |spec|
  spec.name          = "sparrowx-jekyll-theme"
  spec.version       = "0.1.1"
  spec.authors       = ["Mighil Puthukkudy"]
  spec.email         = ["webmaster@mighil.com"]
 
  spec.summary       = %q{Minimal, SEO-friendly, Netlify CMS-ready Jekyll theme. Brother of sparrow by @lingxz.}
  spec.homepage      = "https://github.com/mighildotcom/sparrowx"
  spec.license       = "MIT"
 
  spec.metadata["plugin_type"] = "theme"

  spec.files                   = `git ls-files -z`.split("\x0").select do |f|
    f.match(%r{^(assets/css|assets/fonts|assets/js|_(includes|layouts|sass)/|(LICENSE|README)((\.(txt|md|markdown)|$)))}i)
  end
 
  spec.add_runtime_dependency "jekyll", "~> 3.3"
  spec.add_runtime_dependency "jekyll-paginate", "~> 1.1"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.8"

  spec.add_development_dependency "bundler", "~> 1.12"
end
