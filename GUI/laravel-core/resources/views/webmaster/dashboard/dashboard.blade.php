@extends('layouts.webmaster')
@section('subtitle', "Dashboard")
@section('content')

@php
$user = Auth::user();
@endphp
<div class="row mb-2 mb-xl-3">
	<div class="col-auto d-none d-sm-block">
	  <h3><strong>Fraud</strong> Detection dashboard</h3>
	</div>
  <div class="col-auto ms-auto text-end mt-n1">
	  <!--<a href="#" class="btn btn-light bg-white me-2">Invite a Friend</a>
		<a href="#" class="btn btn-primary">New Project</a>-->
	</div>
</div>
<div class="row">
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Total Fraud Alerts</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="alert-octagon"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">6.132</h1>
				<div class="mb-0">
					<span class="badge badge-success-light">3.65%</span>
					<span class="text-muted">Depuis le mois dernièr</span>
				</div>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Current Active Alerts</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="activity"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">542</h1>
				<div class="mb-0">
					<span class="badge badge-danger-light">-5.25%</span>
					<span class="text-muted">Depuis le mois dernièr</span>
				</div>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Resolved Alerts</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="check-circle"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">5.590</h1>
				<div class="mb-0">
					<span class="badge badge-success-light">4.65%</span>
					<span class="text-muted">Depuis le mois dernièr</span>
				</div>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Alert Severity Levels</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="shield"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">0.65</h1>
        <h1 class="badge badge-danger-light">High</h1>
			</div>
		</div>
	</div>
</div>
<div class="row">
	<div class="col-12 col-lg-4 d-flex">
		<div class="card flex-fill w-100">
			<div class="card-header">
				<div class="float-end">
					<form class="row g-2">
						<div class="col-auto">
							<select class="form-select form-select-sm bg-light border-0">
								<option>2024</option>
							</select>
						</div>
					</form>
				</div>
				<h5 class="card-title mb-0">Fraud Trends Over Time</h5>
			</div>
			<div class="card-body pt-2 pb-3">
				<div class="chart chart-md">
					<canvas id="chartjs-dashboard-line"></canvas>
				</div>
			</div>
		</div>
	</div>
	<div class="col-12 col-lg-4 d-flex">
		<div class="card flex-fill w-100">
			<div class="card-header">
				<div class="float-end">
					<form class="row g-2">
						<div class="col-auto">
							<select class="form-select form-select-sm bg-light border-0">
								<option>2024</option>
							</select>
						</div>
					</form>
				</div>
				<h5 class="card-title mb-0">Geographical Heat Map</h5>
			</div>
			<div class="card-body px-4">
				<div id="algeria_map" style="height:294px;"></div>
			</div>
		</div>
	</div>
  <div class="col-12 col-lg-4 d-flex">
    <div class="card flex-fill w-100">
      <div class="card-header">
        <div class="card-actions float-end">
          <div class="dropdown position-relative">
            <a href="#" data-bs-toggle="dropdown" data-bs-display="static">
              <i class="align-middle" data-feather="more-horizontal"></i>
            </a>  
            <div class="dropdown-menu dropdown-menu-end">
              <a class="dropdown-item" href="#">Action</a>
              <a class="dropdown-item" href="#">Another action</a>
              <a class="dropdown-item" href="#">Something else here</a>
            </div>
          </div>
        </div>
        <h5 class="card-title mb-0">Fraud Types</h5>
      </div>
      <div class="card-body d-flex">
        <div class="align-self-center w-100">
          <div class="py-3">
            <div class="chart chart-xs">
              <canvas id="chartjs-dashboard-pie"></canvas>
            </div>
          </div>  
          <table class="table mb-0">
            <tbody>
              <tr>
                <td><i class="fas fa-circle text-primary fa-fw"></i> Simbox <span
                    class="badge badge-success-light">+12%</span></td>
                <td class="text-end">4306</td>
              </tr>
              <tr>
                <td><i class="fas fa-circle text-warning fa-fw"></i> SIM swapping <span
                    class="badge badge-danger-light">-3%</span></td>
                <td class="text-end">3801</td>
              </tr>
              <tr>
                <td><i class="fas fa-circle text-danger fa-fw"></i> IRSF</td>
                <td class="text-end">1689</td>
              </tr>
              <tr>
                <td><i class="fas fa-circle text-dark fa-fw"></i> Other</td>
                <td class="text-end">3251</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="row">
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">False Positives Rate</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="alert-triangle"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">5.56%</h1>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Detection Rate</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="activity"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">94.95%</h1>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Resolution Time</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="clock"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">8 hours</h1>
			</div>
		</div>
	</div>
	<div class="col-sm-6 col-xl-3">
		<div class="card">
			<div class="card-body">
				<div class="row">
					<div class="col mt-0">
						<h5 class="card-title">Financial Impact</h5>
					</div>
					<div class="col-auto">
						<div class="stat text-primary">
							<i class="align-middle" data-feather="dollar-sign"></i>
						</div>
					</div>
				</div>
				<h1 class="mt-1 mb-3">(+/-) $10,000</h1>
			</div>
		</div>
	</div>
</div>
<div class="row">
	<div class="col-12 col-lg-5 col-xxl d-flex">
		<div class="card flex-fill">
			<div class="card-header">
				<div class="card-actions float-end">
					<button class="btn btn-sm btn-light">View all</button>
				</div>
				<h5 class="card-title mb-0">High-Risk Users</h5>
			</div>
			<table class="table table-sm table-striped my-0">
				<thead>
					<tr>
						<th>Subscription</th>
						<th>User</th>
						<th>Risk rate</th>
					</tr>
				</thead>
				<tbody>
          @foreach($subscriptions as $subscription)
          <tr>
            <td>{{$subscription->phone_number}}</td>
            <td>{{$subscription->Customer()->name}}</td>
            <td>{{$subscription->RiskRate()}}%</td>
          </tr>
          @endforeach
				</tbody>
			</table>
		</div>
	</div>
	<div class="col-12 col-lg-7 col-xxl d-flex">
		<div class="card flex-fill">
			<div class="card-header">
				<div class="card-actions float-end">
					<button class="btn btn-sm btn-light">View all</button>
				</div>
				<h5 class="card-title mb-0">Access Logs</h5>
			</div>
			<table class="table table-sm table-striped my-0">
				<thead>
					<tr>
						<th>Admin</th>
						<th>Section</th>
						<th>Action</th>
						<th>Date</th>
					</tr>
				</thead>
				<tbody>
          @foreach($logs as $log)
          <tr>
            <td>{{$log['user']}}</td>
            <td>{{$log['section']}}</td>
            <td>{{$log['action']}}</td>
            <td>{{$log['date']}}</td>
          </tr>
          @endforeach
				</tbody>
			</table>
		</div>
	</div>
</div>
@endsection
@section("script")
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="{{asset('assets/js/jquery.vmap.min.js')}}"></script>
<script src="{{asset('assets/js/jquery.vmap.algeria.js')}}" charset="utf-8"></script>
<script type="text/javascript">
  jQuery(document).ready(function () {
    jQuery("#algeria_map").vectorMap({
      map: "dz_fr",
      enableZoom: true,
      showTooltip: true,
	  markers: [{
    coords: [0,0],
    name: "Shanghai"
  }],
    });
  });
</script>
<script>
  document.addEventListener("DOMContentLoaded", function() {
    var ctx = document.getElementById("chartjs-dashboard-line").getContext("2d");
    var gradientLight = ctx.createLinearGradient(0, 0, 0, 225);
    gradientLight.addColorStop(0, "rgba(215, 227, 244, 1)");
    gradientLight.addColorStop(1, "rgba(215, 227, 244, 0)");
    var gradientDark = ctx.createLinearGradient(0, 0, 0, 225);
    gradientDark.addColorStop(0, "rgba(51, 66, 84, 1)");
    gradientDark.addColorStop(1, "rgba(51, 66, 84, 0)");
    // Line chart
    new Chart(document.getElementById("chartjs-dashboard-line"), {
      type: "line",
      data: {
        labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        datasets: [{
          label: "Sales ($)",
          fill: true,
          backgroundColor: window.theme.id === "light" ? gradientLight : gradientDark,
          borderColor: window.theme.primary,
          data: [
            2115,
            1562,
            1584,
            1892,
            1587,
            1923,
            2566,
            2448,
            2805,
            3438,
            2917,
            3327
          ]
        }]
      },
      options: {
        maintainAspectRatio: false,
        legend: {
          display: false
        },
        tooltips: {
          intersect: false
        },
        hover: {
          intersect: true
        },
        plugins: {
          filler: {
            propagate: false
          }
        },
        scales: {
          xAxes: [{
            reverse: true,
            gridLines: {
              color: "rgba(0,0,0,0.0)"
            }
          }],
          yAxes: [{
            ticks: {
              stepSize: 1000
            },
            display: true,
            borderDash: [3, 3],
            gridLines: {
              color: "rgba(0,0,0,0.0)",
              fontColor: "#fff"
            }
          }]
        }
      }
    });
  });
</script>
<script>
	document.addEventListener("DOMContentLoaded", function() {
		new Chart(document.getElementById("chartjs-dashboard-pie"), {
			type: "pie",
			data: {
				labels: ["Chrome", "Firefox", "IE", "Other"],
				datasets: [{
					data: [4306, 3801, 1689, 3251],
					backgroundColor: [
						window.theme.primary,
						window.theme.warning,
						window.theme.danger,
						"#E8EAED"
					],
					borderWidth: 5,
					borderColor: window.theme.white
				}]
			},
			options: {
				responsive: !window.MSInputMethodContext,
				maintainAspectRatio: false,
				legend: {
					display: false
				},
				cutoutPercentage: 70
			}
		});
	});
</script>
@endsection