from invoke import task
from datetime import datetime
@task
def git(ctx, message=None):
    """Run the testing script."""
    ctx.run(f"git add .")
    if message is None:
        message = "update"
    ctx.run(f"git commit -m '{message}'")
    ctx.run(f"git push origin main")

@task
def template(ctx):
    """Create a new project from the template."""
    ctx.run("cookiecutter -f --no-input --verbose .")

@task
def requirements(ctx):
    """Install project requirements."""
    ctx.run("python -m pip install --upgrade pip")
    ctx.run("pip install -r requirements.txt")

