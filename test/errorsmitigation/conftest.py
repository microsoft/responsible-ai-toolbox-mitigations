from datetime import datetime
from py.xml import html
import pytest

def pytest_html_report_title(report):
    report.title = "The results for errors mitigation APIs testing"

def pytest_html_results_table_header(cells):
    del cells[1]
    del cells[2]
    cells.insert(1, html.th("API"))
    cells.insert(2, html.th("Test"))
    cells.insert(3, html.th("Duration"))
    cells.insert(4, html.th("Description"))
    cells.pop()


def pytest_html_results_table_row(report, cells):
    del cells[1]
    del cells[2]
    cells.insert(1, html.td(getattr(report, 'api', '')))
    cells.insert(2, html.td(getattr(report, 'test', '')))
    cells.insert(3, html.td(getattr(report, 'duration', '')))
    cells.insert(4, html.td(getattr(report, 'description', '')))
    cells.pop()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()

    
    reportApi = getattr(item.function, '__module__')[5:]
    reportApi = reportApi[:-5]
    report.api = f"{reportApi}"

    reportTest = getattr(item.function, '__name__')[5:]   
    report.test = f"{reportTest}"

    report.description = getattr(item.function, '__doc__')





