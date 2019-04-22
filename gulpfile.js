var gulp = require('gulp');
var minifycss = require('gulp-minify-css');
var uglify = require('gulp-uglify');
var gutil = require('gulp-util');
var htmlmin = require('gulp-htmlmin');
var htmlclean = require('gulp-htmlclean');
var imagemin = require('gulp-imagemin');
// 压缩css文件
gulp.task('minify-css', function() {
  return gulp.src('./public/**/*.css')
  .pipe(minifycss())
  .pipe(gulp.dest('./public'));
});
// 压缩html文件
gulp.task('minify-html', function() {
  return gulp.src('./public/**/*.html')
  .pipe(htmlclean())
  .pipe(htmlmin({
    removeComments: true,
    minifyJS: true,
    minifyCSS: true,
    minifyURLs: true,
  }))
  .pipe(gulp.dest('./public'))
});
// 压缩js文件
// 压缩public目录下的所有js
gulp.task('minify-js', function() {
    return gulp.src('./public/**/*.js')
      .pipe(uglify())
	  .on('error', function (err) { gutil.log(gutil.colors.red('[Error]'), err.toString()); }) //增加这一行
      .pipe(gulp.dest('./public'));
		
});
gulp.task('uglify', function(){
  gulp.src('*.js')
    .pipe(babel({
        presets: ['es2015']
    }))
    .pipe(uglify().on('error', function(e){
        console.log(e);
     }))
    .pipe(gulp.dest('js'));
});
// 压缩 public/demo 目录内图片
gulp.task('minify-images', function() {
    gulp.src('./public/demo/**/*.*')
        .pipe(imagemin({
           optimizationLevel: 5, //类型：Number  默认：3  取值范围：0-7（优化等级）
           progressive: true, //类型：Boolean 默认：false 无损压缩jpg图片
           interlaced: false, //类型：Boolean 默认：false 隔行扫描gif进行渲染
           multipass: false, //类型：Boolean 默认：false 多次优化svg直到完全优化
        }))
        .pipe(gulp.dest('./public/uploads'));
});
// new 兼容
gulp.task('script', function() {
        gulp.src(['public/**/*.js', 'public/lib/**/*.js'])
            .pipe(babel({
                presets: ['es2015'] // es5检查机制
            }))
            .pipe(uglify())
            .on('error', function(err) {
                gutil.log(gutil.colors.red('[Error]'), err.toString());
            })
            .pipe('dist/js')
    });
// 默认任务
gulp.task('default', [
  'minify-html','minify-css','minify-js','minify-images'
]);