# Notes on Reviewing Pull Requests 

A few misc notes on how to review pull requests in github repos. 

Most of this is not specific to `whobpyt`, however you may find alternative options and recommended approaches in other sources. There is always more than one way to skin a git. 


## Making a new PR

(see [here](https://github.com/griffithslab/whobpyt/getting_started/how_to_contribute.md))



## Running PR code locally

To do a comprehensive PR review, you will need to execute the new / modified code to confirm that it is doing what it is suppose to be and not breaking other things. 

The github CI (continuous integration) checks (i.e. the red crosses and green ticks) provide an additional layer of checks of this type, but are not a replacement for locally running PRs before approving a merge.

Here is the process for reviewing a PR locally. It is the same, with perhaps small variations, in any environment you run it (i.e. on your laptop, on a server, on codespaces). 

First - read [this github info page on checking out PRs locally](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/checking-out-pull-requests-locally)

Here is the version of this I (JG) currently use in codespaces (AC edited):

1. Pull the PR

```bash
git fetch upstream pull/<<pull_request_number>>/head:<<pull_request_branch_name>>
git checkout <<pull_request_branch_name>>
```

2. Install whobpyt

```bash
pip install -e .
```

3. Run the new code

```bash
cd examples
python new_example_script.py # For example
```

4. Run the doc build

```bash
cd ..
cd doc
mkdir _static
make clean 
make html
```

5. Review the docs build website

```bash
cd _build/html
python -m http.server
```


