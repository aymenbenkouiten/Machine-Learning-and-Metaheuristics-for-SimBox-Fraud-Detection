<nav id="sidebar" class="sidebar js-sidebar">
  <div class="sidebar-content js-simplebar">
    <a class="sidebar-brand row justify-content-center" href="{{route('dashboard')}}">
      <img src="{{asset('assets/img/icons/djezzy.svg')}}" style="max-height: 100px;" >
    </a>
    <ul class="sidebar-nav">
	  @foreach(config('sidemenu') as $menu)
	  	@if($menu['type'] == "text")
		  @if($user->Has_Permission($menu['permissions']))
          <li class="sidebar-header"> {{$menu['content']}} </li>
		  @endif
		@elseif($menu['type'] == "link")
		  @if($user->Has_Permission($menu['permissions']))
      <li class="sidebar-item {{explode('_', Route::currentRouteName())[0]==$menu['section']?'active':''}}">
        <a class="sidebar-link" href="{{route($menu['route'])}}">
          @if($menu['icon']['type'] == 'feather')
            <i class="align-middle" data-feather="{{$menu['icon']['content']}}"></i>
          @else
            <i class="align-middle me-2 fas fa-fw fa-{{$menu['icon']['content']}}"></i>
          @endif
          <span class="align-middle">{{$menu['content']}}</span>
        </a>
      </li>
      @endif
    @elseif($menu['type'] == 'group')
		  @if($user->Has_Permission($menu['permissions']))
			<li class="sidebar-item {{explode('_', Route::currentRouteName())[0]==$menu['section']?'active':''}}">
				<a data-bs-target="#{{$menu['content']}}" data-bs-toggle="collapse" class="sidebar-link {{explode('_', Route::currentRouteName())[0]==$menu['section']?'':'collapsed'}}">
					
          @if($menu['icon']['type'] == 'feather')
            <i class="align-middle" data-feather="{{$menu['icon']['content']}}"></i>
          @else
            <i class="align-middle me-2 fas fa-fw fa-{{$menu['icon']['content']}}"></i>
          @endif
          <span class="align-middle">{{$menu['content']}}</span>
				</a>
				<ul id="{{$menu['content']}}" class="sidebar-dropdown list-unstyled {{explode('_', Route::currentRouteName())[0]==$menu['section']?'show':'collapse'}}" data-bs-parent="#sidebar">
          @foreach($menu['sub-links'] as $link)
		      @if($user->Has_Permission($link['permissions']))
					<li class="sidebar-item {{Route::currentRouteName()==$link['route']?'active':''}}">
            <a class='sidebar-link' href='{{route($link['route'])}}'>{{$link['content']}}</a>
          </li>
          @endif
          @endforeach
				</ul>
			</li>
      @endif
		@endif
	  @endforeach
    </ul>
  </div>
</nav>