var gulp = require('gulp');
var minifycss = require('gulp-minify-css');
var uglify = require('gulp-uglify');
var gutil = require('gulp-util');
var htmlmin = require('gulp-htmlmin');
var htmlclean = require('gulp-htmlclean');
var imagemin = require('gulp-imagemin');
// ѹ��css�ļ�
gulp.task('minify-css', function() {
  return gulp.src('./public/**/*.css')
  .pipe(minifycss())
  .pipe(gulp.dest('./public'));
});
// ѹ��html�ļ�
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
// ѹ��js�ļ�
// ѹ��publicĿ¼�µ�����js
gulp.task('minify-js', function() {
    return gulp.src('./public/**/*.js')
      .pipe(uglify())
	  .on('error', function (err) { gutil.log(gutil.colors.red('[Error]'), err.toString()); }) //������һ��
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
// ѹ�� public/demo Ŀ¼��ͼƬ
gulp.task('minify-images', function() {
    gulp.src('./public/demo/**/*.*')
        .pipe(imagemin({
           optimizationLevel: 5, //���ͣ�Number  Ĭ�ϣ�3  ȡֵ��Χ��0-7���Ż��ȼ���
           progressive: true, //���ͣ�Boolean Ĭ�ϣ�false ����ѹ��jpgͼƬ
           interlaced: false, //���ͣ�Boolean Ĭ�ϣ�false ����ɨ��gif������Ⱦ
           multipass: false, //���ͣ�Boolean Ĭ�ϣ�false ����Ż�svgֱ����ȫ�Ż�
        }))
        .pipe(gulp.dest('./public/uploads'));
});
// new ����
gulp.task('script', function() {
        gulp.src(['public/**/*.js', 'public/lib/**/*.js'])
            .pipe(babel({
                presets: ['es2015'] // es5������
            }))
            .pipe(uglify())
            .on('error', function(err) {
                gutil.log(gutil.colors.red('[Error]'), err.toString());
            })
            .pipe('dist/js')
    });
// Ĭ������
gulp.task('default', [
  'minify-html','minify-css','minify-js','minify-images'
]);