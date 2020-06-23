
[//]: # (![logo](https://github.com/benmaier/XXXPACKAGENAME/raw/master/img/logo_small.png))

## About

Simulate any epidemiological model on networks or in a well-mixed system.

    git clone git@github.com:benmaier/XXXPACKAGENAME.git
    pip install ./XXXPACKAGENAME

`XXXPACKAGENAME` was developed and tested for 

* Python 3.7

So far, the package's functionality was tested on Mac OS X only.

## Dependencies

`XXXPACKAGENAME` directly depends on the following packages which will be installed by `pip` during the installation process

* `numpy>=1.16`


## Documentation

The full documentation is available at XXX.

## Examples

## Quick API

## Changelog

Changes are logged in a [separate file](https://github.com/benmaier/XXXPACKAGENAME/blob/master/CHANGELOG.md).

## License

This project is licensed under the [MIT License](https://github.com/benmaier/XXXPACKAGENAME/blob/master/LICENSE).

## Contributing

If you want to contribute to this project, please make sure to read the [code of conduct](https://github.com/benmaier/XXXPACKAGENAME/blob/master/CODE_OF_CONDUCT.md) and the [contributing guidelines](https://github.com/benmaier/XXXPACKAGENAME/blob/master/CONTRIBUTING.md). In case you're wondering about what to contribute, we're always collecting ideas of what we want to implement next in the [outlook notes](https://github.com/benmaier/XXXPACKAGENAME/blob/master/OUTLOOK.md).

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-v1.4%20adopted-ff69b4.svg)](code-of-conduct.md)

## Dev notes

Fork this repository, clone it, and install it in dev mode.

```bash
git clone git@github.com:YOURUSERNAME/XXXPACKAGENAME.git
make
```

If you want to upload to PyPI, first convert the new `README.md` to `README.rst`

```bash
make readme
```

It will give you warnings about bad `.rst`-syntax. Fix those errors in `README.rst`. Then wrap the whole thing 

```bash
make pypi
```

It will probably give you more warnings about `.rst`-syntax. Fix those until the warnings disappear. Then do

```bash
make upload
```
