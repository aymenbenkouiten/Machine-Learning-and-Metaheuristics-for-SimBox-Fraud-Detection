@extends('layouts.webmaster')
@section('subtitle', "Settings")
@section('content')
@php
$user = Auth::user();
@endphp
<form action="" method="POST" enctype="multipart/form-data">
@csrf
<div class="row">
  <div class="col-12 col-lg-12 col-xxl-12 d-flex">
    <div class="card flex-fill">
      <div class="card-header d-flex justify-content-between align-items-center">
        <h5 class="card-title mb-0">Settings</h5>
        @if($user->Has_Permission('users_edit'))
		<button type="submit" class="btn btn-primary">Save changes</button>
		@endif
      </div>
    </div>
  </div>
    <div class="col-md-3 col-xl-2">
        <div class="card">
			<div class="card-header">
				<h5 class="card-title mb-0">Settings</h5>
			</div>
			<div class="list-group list-group-flush" role="tablist">
				<a class="list-group-item list-group-item-action active" data-bs-toggle="list" href="#general" role="tab">
					General
				</a>
				@if($user->Has_Permission('facebook_consult'))
				<a class="list-group-item list-group-item-action" data-bs-toggle="list" href="#facebook" role="tab">
					Facebook
				</a>
				@endif
				<a class="list-group-item list-group-item-action" data-bs-toggle="list" href="#notifications" role="tab">
					Notifications
				</a>
				<a class="list-group-item list-group-item-action" data-bs-toggle="list" href="#Triggers" role="tab">
					Triggers
				</a>
			</div>
		</div>
	</div>
	<div class="col-md-9 col-xl-10">
		<div class="tab-content">
			<div class="tab-pane fade show active" id="general" role="tabpanel">
				<div class="card">
					<div class="card-header">
						<h5 class="card-title mb-0">General</h5>
					</div>
					<div class="card-body">
						<div class="row mb-2">
							<div class="col-md-12">
								<div class="mb-3">
									<label class="form-label" for="id">ID</label>
									<input type="text" class="form-control" name="settings-id" id="settings-id" value="{{config('settings.id')}}">
								</div>
								<div class="mb-3">
									<label class="form-label" for="title">Title</label>
									<input type="text" class="form-control" name="settings-title" id="settings-title" value="{{config('settings.title')}}">
								</div>
								<div class="mb-3">
									<label class="form-label" for="title">Conversations limit</label>
									<input type="text" class="form-control" name="settings-limits-conversations" id="settings-limits-conversations" value="{{config('settings.limits.conversations')}}">
								</div>
								<div class="mb-3">
									<label class="form-label" for="title">Message per conversation limit</label>
									<input type="text" class="form-control" name="settings-limits-message_per_conversation" id="settings-limits-message_per_conversation" value="{{config('settings.limits.message_per_conversation')}}">
								</div>
								<div class="mb-3">
									<label class="form-label" for="title">Max simultaneous message</label>
									<input type="text" class="form-control" name="settings-limits-max_simultaneous_message" id="settings-limits-max_simultaneous_message" value="{{config('settings.limits.max_simultaneous_message')}}">
								</div>
								<div class="mb-3">
									<label class="form-label" for="title">Order conversations count</label>
									<input type="text" class="form-control" name="settings-limits-order_conversations_count" id="settings-limits-order_conversations_count" value="{{config('settings.limits.order_conversations_count')}}">
								</div>
							</div>
						</div>
                        @if($user->Has_Permission('settings_edit'))
						<button type="submit" class="btn btn-primary">Save changes</button>
						@endif
					</div>
				</div>
			</div>
			@if($user->Has_Permission('facebook_consult'))
			<div class="tab-pane fade" id="facebook" role="tabpanel">
				<div class="card">
					<div class="card-header">
						<h5 class="card-title mb-0">Facebook</h5>
					</div>
					<div class="card-body mb-2">
						<div class="row mb-2 d-flex align-items-center">
							<div class="col-md-12">
								<div class="mb-3">
									<label for="services-facebook-client_id">Client id</label>
									@if($user->Has_Permission('facebook_edit'))
									<input type="text" class="form-control" name="services-facebook-client_id" id="services-facebook-client_id" value="{{config('services.facebook.client_id')}}">
									@else
									<br>
									<label>{{config('services.facebook.client_id')}}</label>
									@endif
								</div>
								<div class="mb-3">
									<label for="services-facebook-client_secret">Client secret</label>
									@if($user->Has_Permission('facebook_edit'))
									<input type="password" class="form-control" name="services-facebook-client_secret" id="services-facebook-client_secret" value="{{config('services.facebook.client_secret')}}">
									@else
									<br>
									<label>{{config('services.facebook.client_secret')}}</label>
									@endif
								</div>
								<div class="mb-3">
									<label for="services-facebook-redirect">Redirect</label>
									@if($user->Has_Permission('facebook_edit'))
									<input type="text" class="form-control" name="services-facebook-redirect" id="services-facebook-redirect" value="{{config('services.facebook.redirect')}}">
									@else
									<br>
									<label>{{config('services.facebook.redirect')}}</label>
									@endif
								</div>
								@if($user->Has_Permission('facebook_reconnect'))
								<div class="mb-3">
        							<a href="{{route('facebook_reconnect')}}" class="btn btn-facebook"><i class="align-middle fab fa-facebook"></i> Reconnect</a>
								</div>
								{{--
								<div class="mb-3">
        							<a href="{{route('facebook_load_conversations')}}" class="btn btn-facebook"><i class="align-middle fab fa-facebook"></i> {{config('next_page')!=null?"Continue Conversations Loading":"Load All Conversations"}}</a>
								</div>
								--}}
								@endif
							</div>
						</div>
                	@if($user->Has_Permission('settings_edit') && $user->Has_Permission('facebook_edit'))
					<button type="submit" class="btn btn-primary">Save changes</button>
					@endif
					</div>
				</div>
			</div>
			@endif
			<div class="tab-pane fade" id="notifications" role="tabpanel">
				<div class="card">
					<div class="card-header">
						<h5 class="card-title mb-0">Notifications</h5>
					</div>
					<div class="card-body mb-2">
						<div class="row mb-2 d-flex align-items-center">
							<div class="col-md-12">
								<div class="mb-3">
									<label for="username">Username</label>
									<input type="text" class="form-control" name="settings-notifications-username" id="settings-notifications-username" value="{{config('settings.notifications.username')}}">
								</div>
								<div class="mb-3">
									<label for="password">Password</label>
									<input type="password" class="form-control" name="settings-notifications-password" id="settings-notifications-password" value="{{config('settings.notifications.password')}}">
								</div>
								<div class="mb-3">
									<label for="api_token">Api token</label>
									<input type="text" class="form-control" name="settings-notifications-api_token" id="settings-notifications-api_token" value="{{config('settings.notifications.api_token')}}">
								</div>
								<div class="mb-3">
									<label for="package">Package</label>
									<input type="text" class="form-control" name="settings-notifications-package" id="settings-notifications-package" value="{{config('settings.notifications.package')}}">
								</div>
							</div>
						</div>
                	@if($user->Has_Permission('settings_edit'))
					<button type="submit" class="btn btn-primary">Save changes</button>
					@endif
					</div>
				</div>
			</div>
			<div class="tab-pane fade" id="Triggers" role="tabpanel">
				<div class="card">
					<div class="card-header">
						<h5 class="card-title mb-0">Triggers</h5>
					</div>
					<div class="card-body mb-2">
						<div class="row mb-2 d-flex align-items-center">
							<div class="col-md-12">
								<div class="mb-3">
									<div class="form-check form-switch">
										<input class="form-check-input" type="checkbox" id="settings-scheduler-conversations" {{config('settings.scheduler.conversations')?'checked':''}}>
										<input type="hidden" id="settings-scheduler-conversations-value" name="settings-scheduler-conversations" value="{{config('settings.scheduler.conversations')}}">
										<label class="form-check-label" for="settings-scheduler-conversations">Auto update CDR</label>
									</div>
								</div>
								<div class="mb-3">
									<div class="form-check form-switch">
										<input class="form-check-input" type="checkbox" value="true" id="settings-scheduler-orders_states_check" {{config('settings.scheduler.orders_states_check')?'checked':''}}>
										<input type="hidden" id="settings-scheduler-orders_states_check-value" name="settings-scheduler-orders_states_check" value="{{config('settings.scheduler.orders_states_check')}}">
										<label class="form-check-label" for="settings-scheduler-orders_states_check">Auto update data</label>
									</div>
								</div>
								<div class="mb-3">
									<div class="form-check form-switch">
										<input type="hidden" id="settings-scheduler-tokens_validity_check-value" name="settings-scheduler-tokens_validity_check" value="{{config('settings.scheduler.tokens_validity_check')}}">
										<input class="form-check-input" type="checkbox" value="true" id="settings-scheduler-tokens_validity_check" {{config('settings.scheduler.tokens_validity_check')?'checked':''}}>
										<label class="form-check-label" for="settings-scheduler-tokens_validity_check">Auto-Ban when the threshold (+0.95) is crossed</label>
									</div>
								</div>
							</div>
						</div>
                	@if($user->Has_Permission('settings_edit'))
					<button type="submit" class="btn btn-primary">Save changes</button>
					@endif
					</div>
				</div>
			</div>
		</div>
	</div>
</form>
@endsection
@section('script')
<script>
    document.addEventListener('DOMContentLoaded', function() {
        var conversations_checkbox = document.getElementById('settings-scheduler-conversations');
        var conversations_hiddenInput = document.getElementById('settings-scheduler-conversations-value');
        var orders_states_check_checkbox = document.getElementById('settings-scheduler-orders_states_check');
        var orders_states_check_hiddenInput = document.getElementById('settings-scheduler-orders_states_check-value');
        var tokens_validity_check_checkbox = document.getElementById('settings-scheduler-tokens_validity_check');
        var tokens_validity_check_hiddenInput = document.getElementById('settings-scheduler-tokens_validity_check-value');
        var remarketing_send_checkbox = document.getElementById('settings-scheduler-remarketing_send');
        var remarketing_send_hiddenInput = document.getElementById('settings-scheduler-remarketing_send-value');
        var remarketing_interval_send_checkbox = document.getElementById('settings-scheduler-remarketing_interval_send');
        var remarketing_interval_send_hiddenInput = document.getElementById('settings-scheduler-remarketing_interval_send-value');
        var responder_send_checkbox = document.getElementById('settings-scheduler-responder_send');
        var responder_send_hiddenInput = document.getElementById('settings-scheduler-responder_send-value');
        var update_response_time_checkbox = document.getElementById('settings-scheduler-update_response_time');
        var update_response_time_hiddenInput = document.getElementById('settings-scheduler-update_response_time-value');

		
		
        conversations_checkbox.addEventListener('change', function() {
            if (conversations_checkbox.checked) {
                conversations_hiddenInput.value = 1;
            } else {
                conversations_hiddenInput.value = 0;
            }
        });
        orders_states_check_checkbox.addEventListener('change', function() {
            if (orders_states_check_checkbox.checked) {
                orders_states_check_hiddenInput.value = 1;
            } else {
                orders_states_check_hiddenInput.value = 0;
            }
        });
        tokens_validity_check_checkbox.addEventListener('change', function() {
            if (tokens_validity_check_checkbox.checked) {
                tokens_validity_check_hiddenInput.value = 1;
            } else {
                tokens_validity_check_hiddenInput.value = 0;
            }
        });
        remarketing_send_checkbox.addEventListener('change', function() {
            if (remarketing_send_checkbox.checked) {
                remarketing_send_hiddenInput.value = 1;
            } else {
                remarketing_send_hiddenInput.value = 0;
            }
        });
        remarketing_interval_send_checkbox.addEventListener('change', function() {
            if (remarketing_interval_send_checkbox.checked) {
                remarketing_interval_send_hiddenInput.value = 1;
            } else {
                remarketing_interval_send_hiddenInput.value = 0;
            }
        });
        responder_send_checkbox.addEventListener('change', function() {
            if (responder_send_checkbox.checked) {
                responder_send_hiddenInput.value = 1;
            } else {
                responder_send_hiddenInput.value = 0;
            }
        });
        update_response_time_checkbox.addEventListener('change', function() {
            if (update_response_time_checkbox.checked) {
                update_response_time_hiddenInput.value = 1;
            } else {
                update_response_time_hiddenInput.value = 0;
            }
        });
    });
</script>
@endsection